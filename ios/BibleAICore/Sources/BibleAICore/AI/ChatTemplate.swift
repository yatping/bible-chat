import Foundation

/// Pure-Swift renderer for Gemma 4's chat template, used when `llama_chat_apply_template` cannot render the
/// GGUF-embedded template (llama.cpp's C API only pattern-matches a fixed list of templates and, as of b11580,
/// recognises Gemma 2/3's `<start_of_turn>` format but not Gemma 4's).
///
/// Source of truth: the "Google Gemma 4 Canonical Chat Template" (`chat_template.jinja`, published 2026-07-09) in
/// https://huggingface.co/google/gemma-4-E2B-it/blob/main/chat_template.jinja and its `tokenizer_config.json`
/// (`bos_token` = `<bos>`, `eot_token` = `<turn|>`). For text-only messages, no tools and `enable_thinking=false`
/// (what `processor.apply_chat_template` renders in training), the template reduces to:
///
/// ```
/// {bos}<|turn>system\n{system | trim}<turn|>\n
/// <|turn>user\n{content | trim}<turn|>\n
/// <|turn>model\n{strip_thinking(content) | trim}<turn|>\n
/// ...
/// <|turn>model\n                                   (add_generation_prompt)
/// ```
///
/// Roles: `assistant` → `model`. The system turn is only emitted when the first message is a system message.
/// `strip_thinking` removes `<|channel>…<channel|>` thought blocks from model turns.
public enum ChatTemplate {
    public static let bosToken = "<bos>"
    public static let turnStart = "<|turn>"
    public static let turnEnd = "<turn|>"

    /// Strings that end a model turn if they appear in decoded text (safety net next to EOG token detection).
    public static let stopStrings = [turnEnd, turnStart]

    /// Renders `messages` in the Gemma 4 format.
    /// - Parameters:
    ///   - includeBOS: prepend `<bos>`. Pass `false` when the tokenizer adds BOS itself.
    ///   - addGenerationPrompt: end with an open `<|turn>model\n`.
    public static func gemma4(_ messages: [ChatMessage], includeBOS: Bool = false, addGenerationPrompt: Bool = true) -> String {
        var out = includeBOS ? bosToken : ""
        var rest = messages[...]
        if let first = rest.first, first.role == .system {
            out += "\(turnStart)system\n\(trim(first.content))\(turnEnd)\n"
            rest = rest.dropFirst()
        }
        var previousRole: ChatMessage.Role?
        for message in rest {
            switch message.role {
            case .system:
                // Only a leading system message is supported by the template; render later ones as user text.
                out += "\(turnStart)user\n\(trim(message.content))\(turnEnd)\n"
            case .user:
                out += "\(turnStart)user\n\(trim(message.content))\(turnEnd)\n"
            case .assistant:
                // Consecutive assistant messages continue the same model turn.
                if previousRole == .assistant, out.hasSuffix("\(turnEnd)\n") {
                    out.removeLast(turnEnd.count + 1)
                } else {
                    out += "\(turnStart)model\n"
                }
                out += "\(stripThinking(message.content))\(turnEnd)\n"
            }
            previousRole = message.role
        }
        if addGenerationPrompt { out += "\(turnStart)model\n" }
        return out
    }

    /// Port of the template's `strip_thinking` macro.
    static func stripThinking(_ text: String) -> String {
        var result = ""
        for part in text.components(separatedBy: "<channel|>") {
            if let range = part.range(of: "<|channel>") {
                result += part[..<range.lowerBound]
            } else {
                result += part
            }
        }
        return trim(result)
    }

    private static func trim(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
