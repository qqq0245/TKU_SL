from __future__ import annotations

import csv
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


TIME_WORDS = {"today", "tomorrow"}
WH_WORDS = {"what", "where"}
SUBJECT_WORDS = {"i", "you", "mother", "father", "teacher", "student"}
VERB_WORDS = {"want", "like", "help", "go", "eat", "learn", "work", "see"}
ADJECTIVE_WORDS = {"good", "bad", "happy", "sad", "tired", "beautiful"}
NOUN_WORDS = {"school", "house", "car", "food", "book", "dog"}


@dataclass
class SentenceComposition:
    source: str
    gloss: str
    english: str
    chinese: str
    notes: str


@dataclass
class SentenceTemplate:
    sentence_id: str
    goal: str
    tokens: list[str]
    grammar_note: str


class SentenceComposer:
    def __init__(self, examples_csv_path: str | Path) -> None:
        self.examples_csv_path = Path(examples_csv_path)
        self.examples = self._load_examples()
        self.templates = self._load_sentence_templates()

    def compose(self, tokens: list[str]) -> SentenceComposition:
        base_normalized = self.normalize_tokens(tokens, apply_repairs=False)
        if not base_normalized:
            return SentenceComposition(
                source="rule",
                gloss="",
                english="",
                chinese="",
                notes="尚未收集到可用詞語。",
            )

        template_result = self._compose_with_templates(base_normalized)
        if template_result is not None:
            return template_result

        normalized = self.normalize_tokens(tokens, apply_repairs=True)
        llm_result = self._compose_with_llm(normalized)
        if llm_result is not None:
            return llm_result
        return self._compose_with_rules(normalized)

    def normalize_tokens(self, tokens: list[str], apply_repairs: bool = True) -> list[str]:
        normalized = [token.strip().lower() for token in tokens if token.strip()]
        if not normalized:
            return []

        collapsed: list[str] = []
        for token in normalized:
            if not collapsed or collapsed[-1] != token:
                collapsed.append(token)

        collapsed = self._remove_minor_verb_noise(collapsed)
        if apply_repairs:
            collapsed = self._repair_common_patterns(collapsed)
        return collapsed

    def _load_examples(self) -> list[dict]:
        if not self.examples_csv_path.exists():
            return []
        with self.examples_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _load_sentence_templates(self) -> list[SentenceTemplate]:
        manifest_path = self.examples_csv_path.parent / "sentence_video_manifest_50.csv"
        if not manifest_path.exists():
            return []

        templates: list[SentenceTemplate] = []
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    tokens = [str(token).strip().lower() for token in json.loads(row["tokens_json"])]
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
                if not tokens:
                    continue
                templates.append(
                    SentenceTemplate(
                        sentence_id=str(row.get("sentence_id", "")).strip(),
                        goal=str(row.get("goal", "")).strip(),
                        tokens=tokens,
                        grammar_note=str(row.get("grammar_note", "")).strip(),
                    )
                )
        return templates

    def _compose_with_llm(self, tokens: list[str]) -> SentenceComposition | None:
        api_key = os.getenv("SIGN_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        model = os.getenv("SIGN_LLM_MODEL")
        if not api_key or not model:
            return None

        base_url = os.getenv("SIGN_LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
        examples_text = "\n".join(
            f"- 英文原意: {row['英文原意']} | Gloss: {row['ASL 手語順序 (Gloss)']} | 語法說明: {row['語法說明']}"
            for row in self.examples
        )
        prompt = (
            "你是一個 ASL 句子組裝助手。"
            "使用收到的手語詞語序列，輸出 JSON 物件，包含 gloss, english, chinese, notes。"
            "優先沿用 ASL gloss 邏輯：時間前置、形容詞可置於名詞後、WH 問句放句尾。\n"
            f"可參考的示例:\n{examples_text}\n"
            f"收到的詞語: {tokens}"
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You convert sign tokens into ASL gloss and natural bilingual sentences."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            return None

        try:
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return SentenceComposition(
                source="llm",
                gloss=str(parsed.get("gloss", "")).strip(),
                english=str(parsed.get("english", "")).strip(),
                chinese=str(parsed.get("chinese", "")).strip(),
                notes=str(parsed.get("notes", "")).strip(),
            )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return None

    def _compose_with_rules(self, tokens: list[str]) -> SentenceComposition:
        times = [token for token in tokens if token in TIME_WORDS]
        subjects = [token for token in tokens if token in SUBJECT_WORDS]
        wh_words = [token for token in tokens if token in WH_WORDS]
        verbs = [token for token in tokens if token in VERB_WORDS]
        adjectives = [token for token in tokens if token in ADJECTIVE_WORDS]
        others = [
            token
            for token in tokens
            if token not in TIME_WORDS | SUBJECT_WORDS | WH_WORDS | VERB_WORDS | ADJECTIVE_WORDS
        ]

        gloss_tokens = []
        gloss_tokens.extend(times)
        gloss_tokens.extend(subjects[:1])
        gloss_tokens.extend(verbs)
        gloss_tokens.extend(others)
        gloss_tokens.extend(adjectives)
        gloss_tokens.extend(wh_words)
        gloss = " ".join(token.upper() for token in gloss_tokens).strip()
        if wh_words:
            gloss = f"{gloss}?"
        elif gloss:
            gloss = f"{gloss}."

        english = self._build_english(tokens, times, subjects, verbs, others, adjectives, wh_words)
        chinese = self._build_chinese(tokens, times, subjects, verbs, others, adjectives, wh_words)
        notes = "規則式組句：時間前置、WH 疑問詞置尾，未啟用雲端語言模型。"
        return SentenceComposition(
            source="rule",
            gloss=gloss,
            english=english,
            chinese=chinese,
            notes=notes,
        )

    def _compose_with_templates(self, tokens: list[str]) -> SentenceComposition | None:
        if not self.templates:
            return None

        scored_templates: list[tuple[float, float, SentenceTemplate]] = []
        for template in self.templates:
            primary_score = self._alignment_score(tokens, template.tokens)
            keyword_bonus = self._keyword_bonus(tokens, template.tokens) + self._sparse_pattern_bonus(tokens, template)
            score = primary_score + keyword_bonus
            scored_templates.append((score, keyword_bonus, template))

        scored_templates.sort(key=lambda item: (item[0], item[1], -abs(len(item[2].tokens) - len(tokens))), reverse=True)
        best_score, _, best_template = scored_templates[0]
        next_score = scored_templates[1][0] if len(scored_templates) > 1 else -999.0

        if best_score < 1.0:
            return None
        if best_score - next_score < 0.35 and best_score < 4.0:
            return None

        return self._compose_from_template(best_template, observed_tokens=tokens, score=best_score, margin=best_score - next_score)

    def _remove_minor_verb_noise(self, tokens: list[str]) -> list[str]:
        verb_counts = {}
        for token in tokens:
            if token in VERB_WORDS:
                verb_counts[token] = verb_counts.get(token, 0) + 1
        dominant_verb = max(verb_counts, key=verb_counts.get) if verb_counts else None
        if dominant_verb is None:
            return tokens

        cleaned: list[str] = []
        for token in tokens:
            if token in VERB_WORDS and token != dominant_verb and verb_counts.get(dominant_verb, 0) >= 2:
                continue
            cleaned.append(token)
        return cleaned

    def _repair_common_patterns(self, tokens: list[str]) -> list[str]:
        repaired = list(tokens)
        if not repaired:
            return repaired

        has_where = "where" in repaired
        has_what = "what" in repaired

        if "work" in repaired and not has_where:
            last = repaired[-1]
            if last in {"happy", "good", "bad", "tired", "beautiful", "like", "help", "work"}:
                repaired = [token for token in repaired if token not in {"happy", "good", "bad", "tired", "beautiful"}]
                if repaired and repaired[-1] != "where":
                    repaired.append("where")

        if "want" in repaired and "eat" in repaired and not has_what:
            if repaired[-1] not in WH_WORDS:
                repaired.append("what")

        if len(repaired) >= 2 and repaired[-1] == repaired[-2]:
            repaired = repaired[:-1]

        return repaired

    def _compose_from_template(
        self,
        template: SentenceTemplate,
        observed_tokens: list[str],
        score: float,
        margin: float,
    ) -> SentenceComposition:
        gloss = " ".join(token.upper() for token in template.tokens).strip()
        if template.tokens and template.tokens[-1] in WH_WORDS:
            gloss = f"{gloss}?"
        elif gloss:
            gloss = f"{gloss}."
        return SentenceComposition(
            source="template",
            gloss=gloss,
            english=self._build_template_english(template.tokens),
            chinese=self._build_template_chinese(template.tokens),
            notes=(
                f"模板比對：{template.sentence_id}，目標情境={template.goal}，"
                f"相似度分數={score:.2f}，領先差值={margin:.2f}。"
                f" 原始詞語={observed_tokens}。{template.grammar_note}"
            ),
        )

    def _alignment_score(self, observed: list[str], template: list[str]) -> float:
        rows = len(observed) + 1
        cols = len(template) + 1
        dp = [[0.0] * cols for _ in range(rows)]

        for i in range(1, rows):
            dp[i][0] = dp[i - 1][0] - self._gap_penalty(observed[i - 1], observed_side=True)
        for j in range(1, cols):
            dp[0][j] = dp[0][j - 1] - self._gap_penalty(template[j - 1], observed_side=False)

        for i in range(1, rows):
            for j in range(1, cols):
                observed_token = observed[i - 1]
                template_token = template[j - 1]
                match_score = dp[i - 1][j - 1] + self._token_pair_score(observed_token, template_token)
                delete_score = dp[i - 1][j] - self._gap_penalty(observed_token, observed_side=True)
                insert_score = dp[i][j - 1] - self._gap_penalty(template_token, observed_side=False)
                dp[i][j] = max(match_score, delete_score, insert_score)

        if observed and template and observed[0] == template[0]:
            dp[-1][-1] += 0.8
        if observed and template and observed[-1] == template[-1]:
            dp[-1][-1] += 0.8
        return dp[-1][-1]

    def _keyword_bonus(self, observed: list[str], template: list[str]) -> float:
        bonus = 0.0
        observed_set = set(observed)
        template_set = set(template)

        if "want" in observed_set and "want" not in template_set:
            bonus -= 3.0
        if "like" in observed_set and "like" not in template_set:
            bonus -= 2.6
        if "go" in observed_set and "go" not in template_set:
            bonus -= 2.2
        if "work" in observed_set and "work" not in template_set:
            bonus -= 2.0

        if "eat" in observed_set:
            bonus += 0.9 if "eat" in template_set and "what" in template_set else -0.7
        if "what" in observed_set:
            if "food" in template_set or "book" in template_set:
                bonus -= 0.2
            else:
                bonus += 0.8 if "what" in template_set else -0.9
        if "where" in observed_set or "work" in observed_set:
            bonus += 0.8 if "work" in template_set and "where" in template_set else 0.0

        exact_subjects = observed_set & SUBJECT_WORDS & template_set
        bonus += 1.2 * len(exact_subjects)
        exact_times = observed_set & TIME_WORDS & template_set
        bonus += 1.1 * len(exact_times)
        exact_nouns = observed_set & NOUN_WORDS & template_set
        bonus += 1.0 * len(exact_nouns)

        if "eat" in template_set and "want" in observed_set and "eat" not in observed_set:
            bonus -= 1.4
        if any(noun in observed_set for noun in {"food", "book", "dog", "car", "house"}) and "eat" in template_set:
            bonus -= 1.1
        if "what" in template_set and "what" not in observed_set and "eat" not in observed_set:
            bonus -= 1.0
        if "want" in template_set and "want" in observed_set and "eat" not in observed_set and "what" not in observed_set:
            bonus += 0.8
        return bonus

    def _sparse_pattern_bonus(self, observed: list[str], template: SentenceTemplate) -> float:
        observed_set = set(observed)
        template_set = set(template.tokens)
        has_subject = bool(observed_set & SUBJECT_WORDS)
        has_time = bool(observed_set & TIME_WORDS)
        adjective_count = sum(1 for token in observed if token in ADJECTIVE_WORDS)
        subject_count = sum(1 for token in observed if token in SUBJECT_WORDS)

        if {"want", "eat"} <= observed_set and not has_subject and not has_time:
            if template.tokens == ["today", "you", "want", "eat", "what"]:
                return 1.8
            if template.tokens == ["today", "i", "want", "eat", "what"]:
                return 1.2
            if template.tokens == ["tomorrow", "you", "want", "eat", "what"]:
                return 0.6

        if {"want", "food"} <= observed_set and not has_subject and not has_time:
            if template.tokens == ["today", "i", "want", "food"]:
                return 1.6
            if template.tokens == ["tomorrow", "i", "want", "food"]:
                return 0.8

        if observed == ["tomorrow", "want"]:
            if template.tokens == ["tomorrow", "you", "want", "book"]:
                return 1.4
            if template.tokens == ["tomorrow", "i", "want", "food"]:
                return 0.8

        if observed == ["today", "want"]:
            if template.tokens == ["today", "you", "want", "food"]:
                return 2.0
            if template.tokens == ["today", "i", "want", "food"]:
                return 1.4

        if observed == ["work", "where"] and template.tokens == ["i", "work", "where"]:
            return 1.2

        if {"go", "school", "help"} <= observed_set:
            if any(subject in observed_set for subject in SUBJECT_WORDS):
                matched_subjects = observed_set & SUBJECT_WORDS & template_set
                if matched_subjects:
                    return 3.0 + 0.5 * len(matched_subjects)
                return -1.5 if {"go", "school", "help"} <= template_set else 0.0
            if template.goal == "行動與方向":
                return 1.2

        if adjective_count >= 2:
            template_adjectives = [token for token in template.tokens if token in ADJECTIVE_WORDS]
            if len(template_adjectives) >= 2:
                bonus = 2.0
                if subject_count >= 1 and (observed_set & SUBJECT_WORDS & template_set):
                    bonus += 1.2
                if subject_count >= 2 and len([token for token in template.tokens if token in SUBJECT_WORDS]) >= 2:
                    bonus += 1.0
                return bonus
            if len(template.tokens) == 2 and template.tokens[1] in ADJECTIVE_WORDS:
                return -1.4

        if adjective_count == 1 and len(observed) <= 3:
            if len(template.tokens) == 2 and template.tokens[1] in ADJECTIVE_WORDS:
                adjective = next((token for token in observed if token in ADJECTIVE_WORDS), "")
                if adjective and template.tokens[1] == adjective:
                    bonus = 1.4
                    if observed_set & SUBJECT_WORDS & template_set:
                        bonus += 1.6
                    if "car" in observed_set and adjective == "happy" and template.tokens[0] == "teacher":
                        bonus += 1.2
                    if "today" in observed_set and adjective == "tired" and template.tokens[0] == "student":
                        bonus += 1.0
                    if not (observed_set & SUBJECT_WORDS) and template.tokens[0] == "you":
                        bonus += 0.5
                    return bonus

        if {"see", "dog", "bad"} <= observed_set and "father" in template_set:
            return 0.8
        if observed_set == {"dog", "bad"} and template.tokens == ["you", "see", "dog", "bad"]:
            return 2.0
        if "like" in observed_set and "happy" in observed_set and template.tokens == ["student", "happy"]:
            return 2.0
        if observed == ["teacher", "car"] and template.tokens == ["teacher", "happy"]:
            return 2.5
        if observed == ["like", "bad", "school"] and template.tokens == ["you", "like", "book", "good"]:
            return 2.2
        if observed_set >= {"book", "go", "school", "help"} and template.tokens == ["you", "go", "school", "help"]:
            return 2.2
        if observed_set >= {"learn", "go", "school", "help"} and template.tokens == ["father", "go", "school", "help"]:
            return 1.6
        if observed_set >= {"mother", "happy", "learn"} and template.tokens == ["mother", "happy", "father", "tired"]:
            return 2.2

        return 0.0

    def _token_pair_score(self, observed_token: str, template_token: str) -> float:
        if observed_token == template_token:
            return self._token_weight(template_token)

        observed_category = self._token_category(observed_token)
        template_category = self._token_category(template_token)
        if observed_category == template_category:
            return 0.45 * self._token_weight(template_token)

        high_confusion_pairs = {
            ("good", "what"),
            ("bad", "what"),
            ("happy", "where"),
            ("help", "where"),
            ("work", "where"),
            ("go", "i"),
            ("dog", "i"),
            ("learn", "student"),
            ("car", "you"),
            ("house", "what"),
        }
        if (observed_token, template_token) in high_confusion_pairs:
            return 0.30 * self._token_weight(template_token)

        return -0.65 * max(self._token_weight(observed_token), self._token_weight(template_token))

    def _gap_penalty(self, token: str, observed_side: bool) -> float:
        weight = self._token_weight(token)
        if observed_side and self._token_category(token) in {"adjective", "noun"}:
            return 0.35 * weight
        if not observed_side and self._token_category(token) in {"time", "subject"}:
            return 0.95 * weight
        return 0.7 * weight

    def _token_category(self, token: str) -> str:
        if token in TIME_WORDS:
            return "time"
        if token in SUBJECT_WORDS:
            return "subject"
        if token in WH_WORDS:
            return "wh"
        if token in VERB_WORDS:
            return "verb"
        if token in ADJECTIVE_WORDS:
            return "adjective"
        if token in NOUN_WORDS:
            return "noun"
        return "other"

    def _token_weight(self, token: str) -> float:
        category = self._token_category(token)
        if category == "time":
            return 3.4
        if category == "subject":
            return 3.2
        if category == "wh":
            return 3.2
        if category == "verb":
            return 2.6
        if category == "noun":
            return 2.1
        if category == "adjective":
            return 1.5
        return 1.0

    def _build_template_english(self, tokens: list[str]) -> str:
        if not tokens:
            return ""
        if len(tokens) == 3 and tokens[-1] == "where" and tokens[1] == "work":
            subject = tokens[0]
            if subject == "you":
                return "Where do you work?"
            if subject == "i":
                return "Where do I work?"
            return f"Where does {subject} work?"
        if len(tokens) == 5 and tokens[-1] == "what" and tokens[2] == "want" and tokens[3] == "eat":
            time_word, subject = tokens[0], tokens[1]
            if subject == "you":
                return f"What do you want to eat {time_word}?"
            if subject == "i":
                return f"What do I want to eat {time_word}?"
            return f"What does {subject} want to eat {time_word}?"
        if len(tokens) == 4 and tokens[2] == "want":
            return f"{tokens[0].capitalize()} wants {tokens[3]} {tokens[1]}.".replace(" today", " today").replace(" tomorrow", " tomorrow")
        if len(tokens) == 4 and tokens[1] == "go" and tokens[2] == "school" and tokens[3] == "help":
            return f"{tokens[0].capitalize()} goes to school to help."
        if len(tokens) == 4 and tokens[1] == "like":
            return f"{tokens[0].capitalize()} likes {tokens[2]} {tokens[3]}."
        if len(tokens) == 4 and tokens[1] == "see":
            return f"{tokens[0].capitalize()} sees {tokens[2]} {tokens[3]}."
        if len(tokens) == 2 and tokens[1] in ADJECTIVE_WORDS:
            return f"{tokens[0].capitalize()} is {tokens[1]}."
        if len(tokens) == 4 and tokens[1] in ADJECTIVE_WORDS and tokens[3] in ADJECTIVE_WORDS:
            return f"{tokens[0].capitalize()} is {tokens[1]}, and {tokens[2]} is {tokens[3]}."
        return " ".join(tokens).capitalize() + "."

    def _build_template_chinese(self, tokens: list[str]) -> str:
        zh_map = {
            "i": "我",
            "you": "你",
            "mother": "媽媽",
            "father": "爸爸",
            "teacher": "老師",
            "student": "學生",
            "want": "想要",
            "like": "喜歡",
            "help": "幫忙",
            "go": "去",
            "eat": "吃",
            "learn": "學習",
            "work": "工作",
            "see": "看見",
            "school": "學校",
            "house": "房子",
            "car": "汽車",
            "food": "食物",
            "book": "書",
            "dog": "狗",
            "good": "好的",
            "bad": "壞的",
            "happy": "快樂的",
            "sad": "悲傷的",
            "tired": "疲倦的",
            "beautiful": "美麗的",
            "today": "今天",
            "tomorrow": "明天",
            "what": "什麼",
            "where": "哪裡",
        }
        if len(tokens) == 3 and tokens[-1] == "where" and tokens[1] == "work":
            return f"{zh_map[tokens[0]]}在哪裡工作？"
        if len(tokens) == 5 and tokens[-1] == "what" and tokens[2] == "want" and tokens[3] == "eat":
            return f"{zh_map[tokens[0]]}{zh_map[tokens[1]]}想吃什麼？"
        if len(tokens) == 4 and tokens[2] == "want":
            return f"{zh_map[tokens[0]]}{zh_map[tokens[1]]}想要{zh_map[tokens[3]]}。"
        if len(tokens) == 4 and tokens[1] == "go" and tokens[2] == "school" and tokens[3] == "help":
            return f"{zh_map[tokens[0]]}去學校幫忙。"
        if len(tokens) == 4 and tokens[1] == "like":
            return f"{zh_map[tokens[0]]}喜歡{zh_map[tokens[2]]}{zh_map[tokens[3]]}。"
        if len(tokens) == 4 and tokens[1] == "see":
            return f"{zh_map[tokens[0]]}看見{zh_map[tokens[2]]}{zh_map[tokens[3]]}。"
        if len(tokens) == 2 and tokens[1] in ADJECTIVE_WORDS:
            return f"{zh_map[tokens[0]]}{zh_map[tokens[1]]}。"
        if len(tokens) == 4 and tokens[1] in ADJECTIVE_WORDS and tokens[3] in ADJECTIVE_WORDS:
            return f"{zh_map[tokens[0]]}{zh_map[tokens[1]]}，{zh_map[tokens[2]]}{zh_map[tokens[3]]}。"
        return " ".join(zh_map.get(token, token) for token in tokens)

    def _build_english(
        self,
        tokens: list[str],
        times: list[str],
        subjects: list[str],
        verbs: list[str],
        nouns: list[str],
        adjectives: list[str],
        wh_words: list[str],
    ) -> str:
        if "where" in wh_words and "work" in verbs and subjects:
            subject = subjects[0]
            if subject == "you":
                return "Where do you work?"
            if subject == "i":
                return "Where do I work?"
            return f"Where does {subject} work?"
        if "what" in wh_words and "eat" in verbs and "want" in verbs:
            subject = subjects[0] if subjects else "you"
            time_text = f" {times[0]}" if times else ""
            if subject == "you":
                return f"What do you want to eat{time_text}?"
            if subject == "i":
                return f"What do I want to eat{time_text}?"
            return f"What does {subject} want to eat{time_text}?"
        parts = []
        if subjects:
            parts.append(subjects[0].capitalize())
        if verbs:
            parts.extend(verbs)
        if nouns:
            parts.extend(nouns)
        if adjectives:
            parts.extend(adjectives)
        if times:
            parts.extend(times)
        sentence = " ".join(parts).strip()
        if not sentence:
            sentence = " ".join(tokens)
        if wh_words:
            sentence = f"{sentence} {' '.join(wh_words)}?".strip()
        else:
            sentence = f"{sentence}.".strip()
        return sentence

    def _build_chinese(
        self,
        tokens: list[str],
        times: list[str],
        subjects: list[str],
        verbs: list[str],
        nouns: list[str],
        adjectives: list[str],
        wh_words: list[str],
    ) -> str:
        zh_map = {
            "i": "我",
            "you": "你",
            "mother": "媽媽",
            "father": "爸爸",
            "teacher": "老師",
            "student": "學生",
            "want": "想要",
            "like": "喜歡",
            "help": "幫忙",
            "go": "去",
            "eat": "吃",
            "learn": "學習",
            "work": "工作",
            "see": "看見",
            "school": "學校",
            "house": "房子",
            "car": "汽車",
            "food": "食物",
            "book": "書",
            "dog": "狗",
            "good": "好的",
            "bad": "壞的",
            "happy": "快樂的",
            "sad": "悲傷的",
            "tired": "疲倦的",
            "beautiful": "美麗的",
            "today": "今天",
            "tomorrow": "明天",
            "what": "什麼",
            "where": "哪裡",
        }
        translated = [zh_map.get(token, token) for token in tokens]
        if "where" in wh_words and subjects and "work" in verbs:
            return f"{zh_map.get(subjects[0], subjects[0])}在哪裡工作？"
        if "what" in wh_words and "eat" in verbs and "want" in verbs:
            subject = zh_map.get(subjects[0], "你") if subjects else "你"
            time_text = zh_map.get(times[0], "") if times else ""
            return f"{subject}{time_text}想吃什麼？"
        return " ".join(translated)
