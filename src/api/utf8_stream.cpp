#include "vastloom/api/utf8_stream.hpp"

#include <cstddef>

namespace vastloom::api {
namespace {

// Expected total length of a UTF-8 sequence from its lead byte; 0 for a byte
// that cannot start one (a stray continuation byte or an invalid lead).
std::size_t sequence_length(unsigned char lead) {
    if (lead < 0x80U) return 1U;
    if ((lead & 0xE0U) == 0xC0U) return 2U;
    if ((lead & 0xF0U) == 0xE0U) return 3U;
    if ((lead & 0xF8U) == 0xF0U) return 4U;
    return 0U;
}

bool continuation(unsigned char byte) { return (byte & 0xC0U) == 0x80U; }

}  // namespace

std::string_view Utf8StreamFilter::feed(std::string_view piece) {
    ready_.clear();
    std::string combined;
    combined.reserve(pending_.size() + piece.size());
    combined.append(pending_);
    combined.append(piece);
    pending_.clear();

    std::size_t index = 0U;
    while (index < combined.size()) {
        const auto lead = static_cast<unsigned char>(combined[index]);
        const std::size_t length = sequence_length(lead);
        if (length <= 1U) {
            // ASCII, or a byte that cannot start a sequence: forward it and let
            // the JSON serializer deal with invalidity.
            ready_.push_back(combined[index]);
            ++index;
            continue;
        }
        if (index + length > combined.size()) {
            // Tail shorter than a whole sequence: buffer it only if the bytes we
            // already have are continuations, otherwise it is malformed and is
            // forwarded now.
            bool valid_prefix = true;
            for (std::size_t cursor = index + 1U; cursor < combined.size(); ++cursor) {
                if (!continuation(static_cast<unsigned char>(combined[cursor]))) {
                    valid_prefix = false;
                    break;
                }
            }
            if (valid_prefix) {
                pending_.assign(combined, index, combined.size() - index);
            } else {
                ready_.append(combined, index, combined.size() - index);
            }
            index = combined.size();
            break;
        }
        bool valid = true;
        for (std::size_t cursor = 1U; cursor < length; ++cursor) {
            if (!continuation(static_cast<unsigned char>(combined[index + cursor]))) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            // Emit the lead as an invalid byte and resynchronise on the next one.
            ready_.push_back(combined[index]);
            ++index;
            continue;
        }
        ready_.append(combined, index, length);
        index += length;
    }
    return ready_;
}

std::string_view Utf8StreamFilter::flush() {
    ready_.clear();
    ready_.swap(pending_);
    return ready_;
}

}  // namespace vastloom::api
