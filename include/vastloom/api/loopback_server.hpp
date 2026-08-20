#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

#include "vastloom/core/status.hpp"

namespace vastloom::api {

// A single HTTP request as received by the loopback server. The body is already bounded by the
// caller-supplied ceiling before the handler runs, so a request decides at most that much work.
struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

struct HttpResponse {
    int status{200};
    std::string content_type{"application/json"};
    std::string body;
    // When true the server switches to an SSE stream: it writes the event-stream headers, then
    // invokes `produce` with an `emit` callback the handler uses to push each chunk as it is
    // generated. `emit` returns false once the client is gone, so a producer can stop early.
    bool streaming{false};
    std::function<void(const std::function<bool(std::string_view)>& emit)> produce;
};

// Maps one request to one response. Implemented by the CLI (model-backed) and by tests (stub).
using LoopbackHandler = std::function<HttpResponse(const HttpRequest&)>;

// A deliberately small HTTP/1.1 server that binds only to 127.0.0.1. It is single-threaded and
// serves one connection at a time: this is a local endpoint for one interactive client, not a
// public or concurrent server, and keeping it serial keeps its resource envelope bounded and its
// cancellation trivial. Remote access is out of scope by construction - the bind address is fixed.
class LoopbackServer {
public:
    LoopbackServer();
    ~LoopbackServer();

    LoopbackServer(const LoopbackServer&) = delete;
    LoopbackServer& operator=(const LoopbackServer&) = delete;

    // Binds 127.0.0.1:port (port 0 asks the OS for an ephemeral one) and starts accepting on a
    // background thread. `max_body_bytes` bounds every request body before the handler sees it.
    [[nodiscard]] Result<void> start(std::uint16_t port, LoopbackHandler handler,
                                     std::size_t max_body_bytes);

    // The port actually bound (meaningful when start() was given 0).
    [[nodiscard]] std::uint16_t port() const;

    // Stops the accept loop and joins the thread. Idempotent.
    void stop();

    [[nodiscard]] bool running() const;

private:
    struct Impl;
    // Owned outright; the server is created and destroyed by the CLI/tests, never copied.
    void* impl_{nullptr};
};

}  // namespace vastloom::api
