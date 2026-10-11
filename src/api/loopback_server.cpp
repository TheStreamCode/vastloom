#include "vastloom/api/loopback_server.hpp"

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace vastloom::api {

namespace {

#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
void close_socket(socket_t s) { closesocket(s); }
int last_socket_error() { return static_cast<int>(WSAGetLastError()); }
// The send length is int on Winsock and size_t on POSIX.
using send_length_t = int;
#else
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
void close_socket(socket_t s) { ::close(s); }
int last_socket_error() { return errno; }
using send_length_t = std::size_t;
#endif

struct WsaInit {
#ifdef _WIN32
    WsaInit() {
        WSADATA data;
        WSAStartup(MAKEWORD(2, 2), &data);
    }
    ~WsaInit() { WSACleanup(); }
#endif
};

// A local client that goes silent mid-request must not hold the serial server (and therefore
// stop()) hostage: every accepted connection carries a receive deadline, and a recv that hits
// it fails the receive loop that already treats received <= 0 as an aborted request.
void apply_receive_deadline(socket_t client) {
#ifdef _WIN32
    const DWORD timeout_ms = 30000;
    ::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms),
                 sizeof(timeout_ms));
#else
    timeval timeout{30, 0};
    ::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout),
                 sizeof(timeout));
#endif
}

const char* reason_for(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default: return "OK";
    }
}

std::string::size_type find_header_value(const std::string& head, const char* name) {
    // Case-insensitive search for "name:" at a line start.
    std::string lower = head;
    for (auto& c : lower) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    std::string needle = std::string("\n") + name;
    for (auto& c : needle) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    needle += ":";
    auto at = lower.find(needle);
    if (at == std::string::npos) return std::string::npos;
    return at + needle.size();
}

}  // namespace

struct LoopbackServer::Impl {
    socket_t listener{kInvalidSocket};
    std::thread thread;
    std::atomic<bool> stop_flag{false};
    std::atomic<std::uint16_t> bound_port{0};
    LoopbackHandler handler;
    std::size_t max_body_bytes{1U * 1024U * 1024U};

    void serve_one(socket_t client) {
        apply_receive_deadline(client);
        std::string buffer;
        char chunk[4096];
        std::string::size_type header_end = std::string::npos;
        while (header_end == std::string::npos) {
            const auto received = ::recv(client, chunk, sizeof(chunk), 0);
            if (received <= 0) return;
            buffer.append(chunk, static_cast<std::size_t>(received));
            if (buffer.size() > max_body_bytes + 65536U) return;
            header_end = buffer.find("\r\n\r\n");
        }
        const std::string head = buffer.substr(0, header_end);
        HttpRequest request;
        {
            const auto line_end = head.find("\r\n");
            const std::string line = head.substr(0, line_end);
            const auto sp1 = line.find(' ');
            const auto sp2 = line.find(' ', sp1 + 1);
            if (sp1 == std::string::npos || sp2 == std::string::npos) return;
            request.method = line.substr(0, sp1);
            request.path = line.substr(sp1 + 1, sp2 - sp1 - 1);
        }
        std::size_t content_length = 0U;
        if (const auto at = find_header_value(head, "content-length"); at != std::string::npos) {
            content_length = static_cast<std::size_t>(std::strtoull(head.c_str() + at, nullptr, 10));
        }
        if (content_length > max_body_bytes) {
            // Drain (without storing) what the client already committed to send, bounded by the
            // ceiling, so the refusal reaches it instead of a connection reset; then refuse.
            std::size_t drained = buffer.size() - (header_end + 4);
            const std::size_t drain_cap =
                std::min(content_length, max_body_bytes + 65536U);
            while (drained < drain_cap) {
                const auto received = ::recv(client, chunk, sizeof(chunk), 0);
                if (received <= 0) break;
                drained += static_cast<std::size_t>(received);
            }
            HttpResponse too_big;
            too_big.status = 413;
            too_big.body = "request body exceeds the configured ceiling";
            write_response(client, too_big);
            return;
        }
        request.body = buffer.substr(header_end + 4);
        while (request.body.size() < content_length) {
            const auto received = ::recv(client, chunk, sizeof(chunk), 0);
            if (received <= 0) return;
            request.body.append(chunk, static_cast<std::size_t>(received));
        }
        request.body.resize(content_length);

        HttpResponse response = handler ? handler(request) : HttpResponse{};
        if (response.streaming && response.produce) {
            write_stream(client, response);
        } else {
            write_response(client, response);
        }
    }

    static bool send_all(socket_t client, const std::string& out) {
        std::size_t sent = 0U;
        while (sent < out.size()) {
            const auto n = ::send(client, out.data() + sent,
                                 static_cast<send_length_t>(out.size() - sent), 0);
            if (n <= 0) return false;
            sent += static_cast<std::size_t>(n);
        }
        return true;
    }

    static void write_stream(socket_t client, const HttpResponse& response) {
        std::string head = "HTTP/1.1 " + std::to_string(response.status) + " " +
                           reason_for(response.status) + "\r\n";
        head += "Content-Type: text/event-stream\r\n";
        head += "Cache-Control: no-cache\r\n";
        head += "Connection: close\r\n\r\n";
        if (!send_all(client, head)) return;
        response.produce([client](std::string_view piece) -> bool {
            return send_all(client, std::string(piece));
        });
    }

    static void write_response(socket_t client, const HttpResponse& response) {
        std::string out = "HTTP/1.1 " + std::to_string(response.status) + " " +
                          reason_for(response.status) + "\r\n";
        out += "Content-Type: " + response.content_type + "\r\n";
        out += "Content-Length: " + std::to_string(response.body.size()) + "\r\n";
        out += "Connection: close\r\n\r\n";
        out += response.body;
        send_all(client, out);
    }

    void run() {
        while (!stop_flag.load()) {
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(listener, &readfds);
            timeval timeout{0, 100000};  // 100 ms so stop() is observed promptly
            const int ready = ::select(static_cast<int>(listener) + 1, &readfds, nullptr,
                                       nullptr, &timeout);
            if (ready <= 0) continue;
            sockaddr_in addr{};
#ifdef _WIN32
            int addr_len = sizeof(addr);
#else
            socklen_t addr_len = sizeof(addr);
#endif
            socket_t client = ::accept(listener, reinterpret_cast<sockaddr*>(&addr), &addr_len);
            if (client == kInvalidSocket) {
                if (stop_flag.load()) break;
                continue;
            }
            serve_one(client);
            close_socket(client);
        }
    }
};

LoopbackServer::LoopbackServer() = default;

LoopbackServer::~LoopbackServer() { stop(); }

Result<void> LoopbackServer::start(std::uint16_t port, LoopbackHandler handler,
                                   std::size_t max_body_bytes) {
    if (impl_ != nullptr) {
        return make_error(ErrorCode::invalid_argument, "loopback server already started");
    }
    [[maybe_unused]] static WsaInit wsa;
    auto* impl = new Impl();
    impl->handler = std::move(handler);
    impl->max_body_bytes = max_body_bytes;

    impl->listener = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (impl->listener == kInvalidSocket) {
        const int err = last_socket_error();
        delete impl;
        return make_error(ErrorCode::internal_error,
                          "unable to create loopback socket", std::to_string(err));
    }
    const int one = 1;
    ::setsockopt(impl->listener, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one),
                 sizeof(one));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);  // 127.0.0.1 only, by construction
    address.sin_port = ::htons(port);
    if (::bind(impl->listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        const int err = last_socket_error();
        close_socket(impl->listener);
        delete impl;
        return make_error(ErrorCode::internal_error, "unable to bind loopback port",
                          std::to_string(err));
    }
    if (::listen(impl->listener, 4) != 0) {
        const int err = last_socket_error();
        close_socket(impl->listener);
        delete impl;
        return make_error(ErrorCode::internal_error, "unable to listen on loopback port",
                          std::to_string(err));
    }
    sockaddr_in bound{};
#ifdef _WIN32
    int bound_len = sizeof(bound);
#else
    socklen_t bound_len = sizeof(bound);
#endif
    if (::getsockname(impl->listener, reinterpret_cast<sockaddr*>(&bound), &bound_len) == 0) {
        impl->bound_port.store(::ntohs(bound.sin_port));
    }
    impl_ = impl;
    impl->thread = std::thread([impl] { impl->run(); });
    return {};
}

std::uint16_t LoopbackServer::port() const {
    return impl_ != nullptr ? static_cast<Impl*>(impl_)->bound_port.load() : 0U;
}

bool LoopbackServer::running() const {
    return impl_ != nullptr && !static_cast<Impl*>(impl_)->stop_flag.load();
}

void LoopbackServer::stop() {
    if (impl_ == nullptr) return;
    auto* impl = static_cast<Impl*>(impl_);
    impl->stop_flag.store(true);
    if (impl->thread.joinable()) impl->thread.join();
    if (impl->listener != kInvalidSocket) close_socket(impl->listener);
    delete impl;
    impl_ = nullptr;
}

}  // namespace vastloom::api
