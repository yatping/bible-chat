/// Sampling parameters for one generation request (defaults from CONTRACTS §5).
public struct GenerationConfig: Sendable, Hashable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var repeatPenalty: Float
    /// Window of recent tokens the repeat penalty looks at.
    public var repeatLastN: Int
    public var maxTokens: Int
    public var seed: UInt32

    public init(
        temperature: Float = 0.3,
        topP: Float = 0.9,
        topK: Int = 40,
        repeatPenalty: Float = 1.1,
        repeatLastN: Int = 64,
        maxTokens: Int = TokenBudget.default.answer,
        seed: UInt32 = 0xB1B1E
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repeatPenalty = repeatPenalty
        self.repeatLastN = repeatLastN
        self.maxTokens = maxTokens
        self.seed = seed
    }

    /// Answer generation: temperature 0.3, top_p 0.9, top_k 40, repeat penalty 1.1, 1200 new tokens.
    public static let answer = GenerationConfig()

    /// Memory extraction and summarisation: greedy (temperature 0), max 256 tokens.
    public static let deterministic = GenerationConfig(temperature: 0, maxTokens: 256)
}
