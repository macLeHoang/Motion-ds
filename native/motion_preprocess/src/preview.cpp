#include "vms_motion/gpu.hpp"
#include "vms_motion/metadata.hpp"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <tuple>

#include <dlfcn.h>
#include <glib-unix.h>
#include <gst-nvevent.h>
#include <gstnvdsinfer.h>
#include <gstnvdsmeta.h>
#include <nlohmann/json.hpp>
#include <nvdspreprocess_meta.h>
#include <unistd.h>

using namespace vms_motion;

namespace {
struct Options {
    std::vector<std::string> inputs;
    std::string config, output = "motion-preview", identity;
    int gpu = 0, width = 1280, height = 720;
    unsigned max_frames = 0;
    bool verify = false, verify_pixels = false;
};

using Key = std::tuple<unsigned, std::uint64_t, std::uint64_t>;

struct State {
    GMainLoop* loop = nullptr;
    GstElement* pipeline = nullptr;
    unsigned component = 1;
    int capacity = 8;
    Options options;
    std::atomic<bool> failed{false};
    std::atomic<bool> quit_requested{false};
    std::atomic<guint> quit_source{0};
    std::uint64_t frames = 0, active = 0, idle = 0, calibrating = 0, tensors = 0;
    std::mutex mutex;
    std::unique_ptr<Gpu> gpu;
    std::map<Key, std::uint64_t> hashes;
};

GstElement* element(const char* factory, const std::string& name) {
    auto* out = gst_element_factory_make(factory, name.c_str());
    if (!out)
        throw std::runtime_error(std::string("missing GStreamer element: ") + factory);
    return out;
}

void link(GstElement* a, GstElement* b) {
    if (!gst_element_link(a, b))
        throw std::runtime_error("cannot link pipeline elements");
}

Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto value = [&]() {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value for " + a);
            return std::string(argv[++i]);
        };
        if (a == "--input")
            o.inputs.push_back(value());
        else if (a == "--config")
            o.config = value();
        else if (a == "--output")
            o.output = value();
        else if (a == "--identity-engine")
            o.identity = value();
        else if (a == "--gpu")
            o.gpu = std::stoi(value());
        else if (a == "--width")
            o.width = std::stoi(value());
        else if (a == "--height")
            o.height = std::stoi(value());
        else if (a == "--max-frames") {
            int n = std::stoi(value());
            if (n < 0)
                throw std::invalid_argument("negative max-frames");
            o.max_frames = unsigned(n);
        } else if (a == "--verify")
            o.verify = true;
        else if (a == "--verify-pixels") {
            o.verify = true;
            o.verify_pixels = true;
        } else if (a == "--help") {
            std::cout
                << "vms_motion_preview --config FILE --input PATH_OR_URI [--input ...]\n"
                << "  [--output DIR] [--gpu 0] [--width 1280] [--height 720]\n"
                << "  [--max-frames N] [--verify] [--verify-pixels] [--identity-engine FILE]\n";
            std::exit(0);
        } else
            throw std::invalid_argument("unknown option: " + a);
    }
    if (o.config.empty() || o.inputs.empty() || o.width < 1 || o.height < 1 || o.gpu < 0)
        throw std::invalid_argument("provide --config, --input, and positive dimensions");
    if (!o.identity.empty())
        o.verify = true;
    return o;
}

std::string library_path() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&vms_motion_reset_source), &info) || !info.dli_fname)
        throw std::runtime_error("cannot locate preprocessing library");
    return std::filesystem::canonical(info.dli_fname).string();
}

struct TempFiles {
    std::filesystem::path path;

    TempFiles() {
        char name[] = "/tmp/vms-motion-XXXXXX";
        auto* result = mkdtemp(name);
        if (!result)
            throw std::runtime_error("cannot create temporary config directory");
        path = result;
    }

    ~TempFiles() {
        std::error_code e;
        std::filesystem::remove_all(path, e);
    }
};

std::string configure(State& state, const TempFiles& temp) {
    auto* file = g_key_file_new();
    GError* error = nullptr;
    if (!g_key_file_load_from_file(file, state.options.config.c_str(), G_KEY_FILE_NONE, &error)) {
        std::string message = error->message;
        g_error_free(error);
        g_key_file_free(file);
        throw std::runtime_error(message);
    }
    state.component = g_key_file_get_integer(file, "property", "unique-id", nullptr);
    if (!state.component) {
        g_key_file_free(file);
        throw std::invalid_argument("unique-id must be positive");
    }
    gsize n = 0;
    auto* shape = g_key_file_get_integer_list(file, "property", "network-input-shape", &n, &error);
    if (error || n != 4 || shape[0] < int(state.options.inputs.size())) {
        if (error)
            g_error_free(error);
        g_free(shape);
        g_key_file_free(file);
        throw std::invalid_argument("tensor capacity must accommodate all input sources");
    }
    state.capacity = shape[0];
    g_free(shape);
    if (g_key_file_get_boolean(file, "property", "process-on-frame", nullptr) != TRUE) {
        g_key_file_free(file);
        throw std::invalid_argument("sample requires process-on-frame=1");
    }
    gsize groups_count = 0;
    auto** groups = g_key_file_get_groups(file, &groups_count);
    for (gsize i = 0; i < groups_count; ++i)
        if (std::string(groups[i]).rfind("group-", 0) == 0)
            g_key_file_remove_group(file, groups[i], nullptr);
    g_strfreev(groups);
    std::vector<gint> ids;
    for (std::size_t i = 0; i < state.options.inputs.size(); ++i)
        ids.push_back(int(i));
    g_key_file_set_integer_list(file, "group-0", "src-ids", ids.data(), ids.size());
    g_key_file_set_boolean(file, "group-0", "process-on-roi", FALSE);
    g_key_file_set_string(file, "property", "custom-lib-path", library_path().c_str());
    g_key_file_set_string(
        file, "property", "custom-tensor-preparation-function", "CustomTensorPreparation"
    );
    g_key_file_set_integer(file, "property", "gpu-id", state.options.gpu);
    g_key_file_set_integer(file, "user-configs", "gpu-id", state.options.gpu);
    if (!state.options.output.empty())
        g_key_file_set_string(
            file,
            "user-configs",
            "preview-dir",
            std::filesystem::absolute(state.options.output).c_str()
        );
    else
        g_key_file_set_string(file, "user-configs", "preview-dir", "");
    gsize size = 0;
    auto* data = g_key_file_to_data(file, &size, nullptr);
    auto path = temp.path / "preprocess.ini";
    std::ofstream out(path);
    out.write(data, size);
    out.close();
    g_free(data);
    g_key_file_free(file);
    if (!out)
        throw std::runtime_error("cannot write runtime config");
    return path.string();
}

void request_quit(State& state) {
    if (state.quit_requested.exchange(true))
        return;
    // Queue through GLib, including when a streaming thread finishes before run() starts.
    state.quit_source = g_idle_add_full(
        G_PRIORITY_DEFAULT,
        [](gpointer loop) -> gboolean {
            g_main_loop_quit(static_cast<GMainLoop*>(loop));
            return G_SOURCE_REMOVE;
        },
        g_main_loop_ref(state.loop),
        [](gpointer loop) {
            g_main_loop_unref(static_cast<GMainLoop*>(loop));
        }
    );
}

void fail(State& state, const std::string& message) {
    std::cerr << "preview verification: " << message << '\n';
    state.failed = true;
    request_quit(state);
}

std::uint64_t hash(const std::vector<unsigned char>& bytes) {
    std::uint64_t out = 14695981039346656037ULL;
    for (auto b : bytes) {
        out ^= b;
        out *= 1099511628211ULL;
    }
    return out;
}

GstPadProbeReturn checksum(GstPad*, GstPadProbeInfo* info, gpointer user) {
    auto& s = *static_cast<State*>(user);
    auto* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    auto* batch = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch)
        return GST_PAD_PROBE_OK;
    std::lock_guard<std::mutex> lock(s.mutex);
    GstMapInfo map = GST_MAP_INFO_INIT;
    try {
        cuda_check(cudaSetDevice(s.options.gpu), "select verification GPU");
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ))
            throw std::runtime_error("checksum buffer mapping failed");
        auto* surface = reinterpret_cast<NvBufSurface*>(map.data);
        for (auto* l = batch->frame_meta_list; l; l = l->next) {
            auto* frame = static_cast<NvDsFrameMeta*>(l->data);
            SurfaceView view(surface, frame->batch_id);
            s.hashes[{frame->source_id, frame->frame_num, frame->buf_pts}] =
                hash(s.gpu->rgba(view.get()));
        }
        gst_buffer_unmap(buffer, &map);
    } catch (const std::exception& e) {
        if (map.data)
            gst_buffer_unmap(buffer, &map);
        fail(s, e.what());
        return GST_PAD_PROBE_DROP;
    }
    return GST_PAD_PROBE_OK;
}

unsigned output_count(NvDsUserMetaList* list) {
    unsigned n = 0;
    for (auto* l = list; l; l = l->next) {
        auto* meta = static_cast<NvDsUserMeta*>(l->data);
        if (meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
            auto* tensor = static_cast<NvDsInferTensorMeta*>(meta->user_meta_data);
            if (tensor && tensor->unique_id == 2)
                ++n;
        }
    }
    return n;
}

GstPadProbeReturn inspect(GstPad*, GstPadProbeInfo* info, gpointer user) {
    auto& s = *static_cast<State*>(user);
    auto* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    auto* batch = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch) {
        fail(s, "batch metadata missing");
        return GST_PAD_PROBE_DROP;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    GstMapInfo mapped = GST_MAP_INFO_INIT;
    try {
        unsigned active = 0, tensor_samples = 0, identity_outputs = 0;
        std::map<NvDsFrameMeta*, int> slots;
        std::set<int> used_slots;
        for (auto* l = batch->frame_meta_list; l; l = l->next) {
            auto* frame = static_cast<NvDsFrameMeta*>(l->data);
            auto* plan = frame_plan(frame, s.component);
            if (!plan)
                throw std::runtime_error("frame placement metadata missing");
            ++s.frames;
            if (plan->status == Status::Active) {
                ++active;
                ++s.active;
                slots[frame] = plan->tensor_slot;
                if (plan->tensor_slot < 0 || plan->plan.tiles.empty() ||
                    !used_slots.insert(plan->tensor_slot).second)
                    throw std::runtime_error("invalid active tensor slot");
            } else {
                if (plan->tensor_slot != -1 || !plan->plan.tiles.empty())
                    throw std::runtime_error("idle frame contains inference data");
                if (plan->status == Status::Idle)
                    ++s.idle;
                else
                    ++s.calibrating;
            }
            identity_outputs += output_count(frame->frame_user_meta_list);
        }
        for (auto* l = batch->batch_user_meta_list; l; l = l->next) {
            auto* user_meta = static_cast<NvDsUserMeta*>(l->data);
            if (user_meta->base_meta.meta_type != NVDS_PREPROCESS_BATCH_META)
                continue;
            auto* pre = static_cast<GstNvDsPreProcessBatchMeta*>(user_meta->user_meta_data);
            if (!pre || !pre->tensor_meta)
                throw std::runtime_error("null tensor metadata was not stripped");
            auto& shape = pre->tensor_meta->tensor_shape;
            if (shape.size() != 4 || shape[0] != int(pre->roi_vector.size()) || shape[1] != 3)
                throw std::runtime_error("tensor shape/ROI count mismatch");
            tensor_samples += shape[0];
            for (std::size_t i = 0; i < pre->roi_vector.size(); ++i) {
                auto& roi = pre->roi_vector[i];
                auto it = slots.find(roi.frame_meta);
                if (it == slots.end() || it->second != int(i))
                    throw std::runtime_error("tensor references an idle frame or wrong slot");
                identity_outputs += output_count(roi.roi_user_meta_list);
            }
        }
        if (active != tensor_samples)
            throw std::runtime_error("active frames and tensor samples differ");
        s.tensors += tensor_samples;
        if (!s.options.identity.empty() && identity_outputs != active)
            throw std::runtime_error("identity inference processed wrong number of samples");
        if (s.options.verify_pixels) {
            cuda_check(cudaSetDevice(s.options.gpu), "select verification GPU");
            if (!gst_buffer_map(buffer, &mapped, GST_MAP_READ))
                throw std::runtime_error("verification mapping failed");
            auto* surface = reinterpret_cast<NvBufSurface*>(mapped.data);
            for (auto* l = batch->frame_meta_list; l; l = l->next) {
                auto* f = static_cast<NvDsFrameMeta*>(l->data);
                Key key{f->source_id, f->frame_num, f->buf_pts};
                auto it = s.hashes.find(key);
                SurfaceView view(surface, f->batch_id);
                if (it == s.hashes.end() || it->second != hash(s.gpu->rgba(view.get())))
                    throw std::runtime_error("original video pixels/timestamps changed");
                s.hashes.erase(it);
            }
            gst_buffer_unmap(buffer, &mapped);
            mapped.data = nullptr;
        }
        if (s.options.max_frames && s.frames >= s.options.max_frames)
            request_quit(s);
    } catch (const std::exception& e) {
        if (mapped.data)
            gst_buffer_unmap(buffer, &mapped);
        fail(s, e.what());
        return GST_PAD_PROBE_DROP;
    }
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn source_event(GstPad*, GstPadProbeInfo* info, gpointer user) {
    auto& s = *static_cast<State*>(user);
    auto* event = GST_PAD_PROBE_INFO_EVENT(info);
    guint source = 0;
    if (GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_DELETED) {
        gst_nvevent_parse_pad_deleted(event, &source);
        vms_motion_reset_source(s.component, source);
    } else if (GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS) {
        gst_nvevent_parse_stream_eos(event, &source);
        vms_motion_reset_source(s.component, source);
    } else if (GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_RESET) {
        gst_nvevent_parse_stream_reset(event, &source);
        vms_motion_reset_source(s.component, source);
    }
    return GST_PAD_PROBE_OK;
}

gboolean bus_message(GstBus*, GstMessage* message, gpointer user) {
    auto& s = *static_cast<State*>(user);
    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
        GError* error = nullptr;
        gchar* detail = nullptr;
        gst_message_parse_error(message, &error, &detail);
        std::cerr << error->message << '\n';
        if (detail)
            std::cerr << detail << '\n';
        g_error_free(error);
        g_free(detail);
        s.failed = true;
        request_quit(s);
    } else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS)
        request_quit(s);
    return G_SOURCE_CONTINUE;
}

void source_pad(GstElement*, GstPad* pad, gpointer sink_element) {
    auto* caps = gst_pad_get_current_caps(pad);
    if (!caps)
        caps = gst_pad_query_caps(pad, nullptr);
    bool video =
        caps && gst_caps_get_size(caps) > 0 &&
        g_str_has_prefix(gst_structure_get_name(gst_caps_get_structure(caps, 0)), "video/");
    if (caps)
        gst_caps_unref(caps);
    if (!video)
        return;
    auto* sink = gst_element_get_static_pad(GST_ELEMENT(sink_element), "sink");
    if (!gst_pad_is_linked(sink) && gst_pad_link(pad, sink) != GST_PAD_LINK_OK) {
        GST_ELEMENT_ERROR(
            GST_ELEMENT(sink_element), STREAM, FAILED, ("Cannot link decoded source"), (nullptr)
        );
    }
    gst_object_unref(sink);
}

std::string uri(const std::string& input) {
    if (gst_uri_is_valid(input.c_str()))
        return input;
    GError* error = nullptr;
    auto* result = gst_filename_to_uri(std::filesystem::absolute(input).c_str(), &error);
    if (!result) {
        std::string message = error->message;
        g_error_free(error);
        throw std::invalid_argument(message);
    }
    std::string value = result;
    g_free(result);
    return value;
}

std::string identity_config(const State& s, const TempFiles& temp) {
    auto path = temp.path / "identity.ini";
    auto parser = std::filesystem::path(library_path()).parent_path() / "libvms_identity_parser.so";
    if (!std::filesystem::exists(parser))
        throw std::runtime_error("build with BUILD_TESTING=ON to use identity verification");
    std::ofstream out(path);
    out << "[property]\ngpu-id=" << s.options.gpu
        << "\nmodel-engine-file=" << std::filesystem::absolute(s.options.identity).string()
        << "\nbatch-size=" << s.capacity
        << "\ngie-unique-id=2\nnetwork-type=0\nnum-detected-classes=1\ninput-tensor-from-meta=1\noutput-tensor-meta=1\n"
        << "parse-bbox-func-name=VmsIdentityParse\ncustom-lib-path=" << parser.string()
        << "\ncluster-mode=4\n";
    out.close();
    if (!out)
        throw std::runtime_error("cannot write identity config");
    return path.string();
}
}

int main(int argc, char** argv) {
    State s;
    std::unique_ptr<TempFiles> temp;
    guint watch = 0, interrupt = 0, terminate = 0;
    try {
        s.options = options(argc, argv);
        gst_init(nullptr, nullptr);
        temp = std::make_unique<TempFiles>();
        if (const char* newmux = std::getenv("USE_NEW_NVSTREAMMUX");
            newmux && std::string(newmux) == "yes")
            throw std::invalid_argument(
                "sample targets the DeepStream 7.1 legacy mux; unset USE_NEW_NVSTREAMMUX"
            );
        s.loop = g_main_loop_new(nullptr, FALSE);
        s.pipeline = gst_pipeline_new("motion-preview");
        auto config = configure(s, *temp);
        auto *mux = element("nvstreammux", "mux"), *pre = element("nvdspreprocess", "preprocess"),
             *sink = element("fakesink", "sink");
        g_object_set(
            mux,
            "batch-size",
            unsigned(s.options.inputs.size()),
            "width",
            s.options.width,
            "height",
            s.options.height,
            "gpu-id",
            s.options.gpu,
            "batched-push-timeout",
            40000,
            "live-source",
            TRUE,
            nullptr
        );
        bool jetson = std::filesystem::exists("/etc/nv_tegra_release");
        g_object_set(mux, "nvbuf-memory-type", jetson ? 4 : 2, nullptr);
        g_object_set(pre, "config-file", config.c_str(), nullptr);
        g_object_set(sink, "sync", FALSE, "async", FALSE, nullptr);
        gst_bin_add_many(GST_BIN(s.pipeline), mux, pre, sink, nullptr);
        link(mux, pre);
        auto* pre_src = gst_element_get_static_pad(pre, "src");
        gst_pad_add_probe(
            pre_src,
            GST_PAD_PROBE_TYPE_BUFFER,
            vms_motion_strip_empty_tensors,
            GUINT_TO_POINTER(s.component),
            nullptr
        );
        gst_object_unref(pre_src);
        auto* pre_sink = gst_element_get_static_pad(pre, "sink");
        gst_pad_add_probe(pre_sink, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, source_event, &s, nullptr);
        if (s.options.verify_pixels) {
            cuda_check(cudaSetDevice(s.options.gpu), "select verification GPU");
            s.gpu = std::make_unique<Gpu>();
            gst_pad_add_probe(pre_sink, GST_PAD_PROBE_TYPE_BUFFER, checksum, &s, nullptr);
        }
        gst_object_unref(pre_sink);
        if (s.options.identity.empty())
            link(pre, sink);
        else {
            auto* infer = element("nvinfer", "identity-infer");
            auto conf = identity_config(s, *temp);
            g_object_set(infer, "config-file-path", conf.c_str(), nullptr);
            gst_bin_add(GST_BIN(s.pipeline), infer);
            link(pre, infer);
            link(infer, sink);
        }
        auto* inspect_pad = gst_element_get_static_pad(sink, "sink");
        gst_pad_add_probe(inspect_pad, GST_PAD_PROBE_TYPE_BUFFER, inspect, &s, nullptr);
        gst_object_unref(inspect_pad);
        for (std::size_t i = 0; i < s.options.inputs.size(); ++i) {
            auto suffix = std::to_string(i);
            auto *source = element("nvurisrcbin", "source_" + suffix),
                 *queue = element("queue", "queue_" + suffix);
            auto *convert = element("nvvideoconvert", "convert_" + suffix),
                 *filter = element("capsfilter", "rgba_" + suffix);
            auto source_uri = uri(s.options.inputs[i]);
            g_object_set(
                source,
                "uri",
                source_uri.c_str(),
                "gpu-id",
                s.options.gpu,
                "disable-audio",
                TRUE,
                nullptr
            );
            g_object_set(
                convert, "gpu-id", s.options.gpu, "nvbuf-memory-type", jetson ? 0 : 2, nullptr
            );
            auto* caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
            g_object_set(filter, "caps", caps, nullptr);
            gst_caps_unref(caps);
            gst_bin_add_many(GST_BIN(s.pipeline), source, queue, convert, filter, nullptr);
            link(queue, convert);
            link(convert, filter);
            g_signal_connect(source, "pad-added", G_CALLBACK(source_pad), queue);
            auto* srcpad = gst_element_get_static_pad(filter, "src");
            auto name = "sink_" + suffix;
            auto* muxpad = gst_element_request_pad_simple(mux, name.c_str());
            if (!muxpad || gst_pad_link(srcpad, muxpad) != GST_PAD_LINK_OK) {
                gst_object_unref(srcpad);
                if (muxpad)
                    gst_object_unref(muxpad);
                throw std::runtime_error("cannot link source to mux");
            }
            gst_object_unref(srcpad);
            gst_object_unref(muxpad);
        }
        auto* bus = gst_element_get_bus(s.pipeline);
        watch = gst_bus_add_watch(bus, bus_message, &s);
        gst_object_unref(bus);
        auto stop = [](gpointer data) -> gboolean {
            request_quit(*static_cast<State*>(data));
            return G_SOURCE_CONTINUE;
        };
        interrupt = g_unix_signal_add(SIGINT, stop, &s);
        terminate = g_unix_signal_add(SIGTERM, stop, &s);
        if (gst_element_set_state(s.pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
            throw std::runtime_error("pipeline startup failed");
        g_main_loop_run(s.loop);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        s.failed = true;
    }
    if (s.pipeline) {
        gst_element_set_state(s.pipeline, GST_STATE_NULL);
        gst_object_unref(s.pipeline);
    }
    for (auto id : {watch, interrupt, terminate, s.quit_source.load()})
        if (id && g_main_context_find_source_by_id(nullptr, id))
            g_source_remove(id);
    if (s.loop)
        g_main_loop_unref(s.loop);
    if (!s.frames) {
        std::cerr << "No video frames were processed\n";
        s.failed = true;
    }
    std::cout << nlohmann::json(
                     {{"frames", s.frames},
                      {"active", s.active},
                      {"idle", s.idle},
                      {"calibrating", s.calibrating},
                      {"tensor_samples", s.tensors},
                      {"identity_verified", !s.options.identity.empty() && !s.failed.load()},
                      {"failed", s.failed.load()}}
                 ).dump()
              << '\n';
    return s.failed ? 1 : 0;
}
