import os
import pickle
import numpy as np

DB_FILE = "reid_db.pkl"


class ReIDDatabase:
    def __init__(self, max_embeddings=20, threshold=0.65):
        self.max_embeddings = max_embeddings
        self.threshold = threshold

        if os.path.exists(DB_FILE):
            with open(DB_FILE, "rb") as f:
                self.db = pickle.load(f)
        else:
            self.db = {}

    def _norm(self, x):
        return x / (np.linalg.norm(x) + 1e-12)

    def _cosine(self, a, b):
        return float(np.dot(a, b))

    def match(self, embedding):
        emb = self._norm(embedding)

        best_match = None
        best_score = -1

        for name, data in self.db.items():
            embs = data["embeddings"]

            scores = [self._cosine(self._norm(e), emb) for e in embs]
            if not scores:
                continue

            max_score = max(scores)
            avg_score = sum(scores) / len(scores)

            score = 0.7 * max_score + 0.3 * avg_score

            if score > best_score:
                best_score = score
                best_match = name

        if best_score >= self.threshold:
            self.db[best_match]["count"] += 1
            return best_match, best_score

        return None, best_score

    def add_unknown(self, embedding):
        emb = self._norm(embedding)

        key = f"unknown_{len(self.db)}"
        self.db[key] = {
            "embeddings": [emb],
            "count": 1
        }
        return key

    
    def update(self, name, embedding):
        emb = self._norm(embedding)

        if name not in self.db:
            return

        self.db[name]["embeddings"].append(emb)

        if len(self.db[name]["embeddings"]) > self.max_embeddings:
            self.db[name]["embeddings"].pop(0)

        self.db[name]["count"] += 1

    
    def merge_similar(self, merge_threshold=0.75):
        names = list(self.db.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]

                if n1 not in self.db or n2 not in self.db:
                    continue

                emb1 = self.db[n1]["embeddings"]
                emb2 = self.db[n2]["embeddings"]

                avg1 = self._norm(np.mean(emb1, axis=0))
                avg2 = self._norm(np.mean(emb2, axis=0))

                score = self._cosine(avg1, avg2)

                if score > merge_threshold:
                    self.db[n1]["embeddings"].extend(self.db[n2]["embeddings"])
                    self.db[n1]["count"] += self.db[n2]["count"]

                    if len(self.db[n1]["embeddings"]) > self.max_embeddings:
                        self.db[n1]["embeddings"] = self.db[n1]["embeddings"][-self.max_embeddings:]

                    del self.db[n2]

    def save(self):
        with open(DB_FILE, "wb") as f:
            pickle.dump(self.db, f)


    def stats(self):
        return {
            "total_identities": len(self.db),
            "details": {
                name: {
                    "embeddings": len(data["embeddings"]),
                    "count": data["count"]
                }
                for name, data in self.db.items()
            }
        }
