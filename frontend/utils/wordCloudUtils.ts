/**
 * Word cloud utilities: tokenize, count frequencies.
 *
 * Flow (same idea as Jason Davies / d3-cloud word clouds):
 * 1. Tokenize: split text into words (lowercase, letters+numbers).
 * 2. Stopword filter: exclude common function words (to, for, from, and, the, is, etc.)
 *    so they do NOT appear in the cloud — only content words (nouns, verbs, etc.) show.
 * 3. Count and sort by frequency; return top N for layout.
 *
 * So "normal" words like "to", "for", "from", "and" are intentionally filtered out.
 */
const STOPWORDS = new Set([
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
  "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
  "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
  "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
  "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
  "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
  "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
  "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
  "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
  "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
  "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
  "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would",
  "you", "your", "yours", "yourself", "yourselves",
]);

const MIN_WORD_LENGTH = 2;
const MAX_WORDS_FOR_CLOUD = 50;

export interface WordCount {
  word: string;
  count: number;
}

/**
 * Tokenize text into words (letters + numbers only), lowercase.
 * Exclude only the stopword list above; include all other words. Count and return top by frequency.
 */
export function getWordCounts(text: string): WordCount[] {
  if (!text || !text.trim()) return [];
  const lower = text.toLowerCase();
  const tokens = lower.match(/[a-z0-9]+/g) || [];
  const counts = new Map<string, number>();
  for (const t of tokens) {
    if (t.length < MIN_WORD_LENGTH) continue;
    if (STOPWORDS.has(t)) continue;
    counts.set(t, (counts.get(t) ?? 0) + 1);
  }
  const sorted = Array.from(counts.entries())
    .map(([word, count]) => ({ word, count }))
    .sort((a, b) => b.count - a.count);
  return sorted.slice(0, MAX_WORDS_FOR_CLOUD);
}
