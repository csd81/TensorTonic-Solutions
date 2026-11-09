def spell_correction(misspelled: list, dictionary: list, frequencies: dict) -> list:
    """
    Returns one corrected dictionary word per input word based on the smallest 
    Levenshtein distance, higher dictionary frequency, and alphabetical tie-breaker.
    """
    def get_levenshtein_distance(w1: str, w2: str) -> int:
        m, n = len(w1), len(w2)
        # Standard Levenshtein DP array
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if w1[i - 1] == w2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],     # Deletion
                        dp[i][j - 1],     # Insertion
                        dp[i - 1][j - 1]  # Substitution
                    )
        return dp[m][n]

    corrected_words = []
    
    for m_word in misspelled:
        # Min function evaluates the sorting keys sequentially:
        # 1. Levenshtein Distance (Ascending: lower is closer)
        # 2. -Frequency (Descending: higher frequency yields smaller value)
        # 3. Word string (Ascending: Python alphabetical order)
        best_match = min(
            dictionary,
            key=lambda d_word: (
                get_levenshtein_distance(m_word, d_word),
                -frequencies.get(d_word, 0),
                d_word
            )
        )
        corrected_words.append(best_match)
        
    return corrected_words
