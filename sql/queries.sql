-- Q1: class distribution with human-readable names
SELECT c.class_name,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM tweets), 2) AS pct
FROM tweets t
JOIN classes c ON t.class_id = c.class_id
GROUP BY c.class_name
ORDER BY n DESC;

-- Q2: average tweet length (chars) per class
SELECT c.class_name,
       ROUND(AVG(LENGTH(t.text)), 1) AS avg_chars,
       MAX(LENGTH(t.text)) AS max_chars
FROM tweets t
JOIN classes c ON t.class_id = c.class_id
GROUP BY c.class_name
ORDER BY avg_chars DESC;

-- Q3: annotator agreement per class (share of tweets where the winning label was unanimous)
SELECT c.class_name,
       ROUND(AVG(a.hate_votes), 2)      AS avg_hate_votes,
       ROUND(AVG(a.offensive_votes), 2) AS avg_offensive_votes,
       ROUND(AVG(a.neither_votes), 2)   AS avg_neither_votes
FROM annotations a
JOIN tweets t ON a.tweet_id = t.tweet_id
JOIN classes c ON t.class_id = c.class_id
GROUP BY c.class_name;

-- Q4: how many annotators rated each tweet, and how the class mix shifts with panel size
SELECT t.annotator_count,
       COUNT(*) AS n_tweets,
       SUM(CASE WHEN t.class_id = 0 THEN 1 ELSE 0 END) AS hate,
       SUM(CASE WHEN t.class_id = 1 THEN 1 ELSE 0 END) AS offensive,
       SUM(CASE WHEN t.class_id = 2 THEN 1 ELSE 0 END) AS neither
FROM tweets t
GROUP BY t.annotator_count
ORDER BY t.annotator_count;

-- Q5: borderline hate-speech tweets (labelled offensive but with at least one hate vote)
SELECT COUNT(*) AS contested_offensive
FROM tweets t
JOIN annotations a ON t.tweet_id = a.tweet_id
WHERE t.class_id = 1 AND a.hate_votes > 0;

-- Q6: ten tweets with the strongest hate-speech consensus
SELECT t.tweet_id, a.hate_votes, t.annotator_count, SUBSTR(t.text, 1, 60) AS snippet
FROM tweets t
JOIN annotations a ON t.tweet_id = a.tweet_id
WHERE t.class_id = 0
ORDER BY a.hate_votes DESC, t.annotator_count DESC
LIMIT 10;
