#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace vastloom::api {

// Buffers a trailing, well-formed but incomplete UTF-8 sequence between stream
// pieces, so a token boundary can never split a code point into two SSE chunks.
// Complete sequences pass through unchanged; invalid bytes are passed through
// unchanged as well (the JSON serializer replaces them with U+FFFD), because this
// filter's only job is to avoid splitting a valid sequence. Feed it before the
// stop-sequence filter so delimiter matching always sees complete code points.
class Utf8StreamFilter {
public:
    // Returns the bytes that are safe to emit now. The returned view is valid
    // until the next call on this filter.
    [[nodiscard]] std::string_view feed(std::string_view piece);

    // Releases the residual incomplete sequence at end of generation. A well
    // formed document has nothing to release here.
    [[nodiscard]] std::string_view flush();

private:
    std::string pending_;
    std::string ready_;
};

}  // namespace vastloom::api
