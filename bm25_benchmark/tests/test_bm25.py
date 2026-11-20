import json
import math
import unittest
from dataclasses import replace
from io import BytesIO
from unittest.mock import patch

from cbr_bm25.benchmark import score_query
from cbr_bm25.data import CORPUS_PATH, Passage, load_corpus
from cbr_bm25.search import BM25Index, SearchIndex, make_tokenizer
from demo.answer import COARENA_FREE_MODELS, INSUFFICIENT, answer_question, chat_completion


def passage(document_id: str, body: str, number: int = 1) -> Passage:
    return Passage(document_id, f"{document_id}#{number:02d}", "Справка",
                   f"https://example.test/{document_id}", "faq", "Условие", body, ())


class BM25Tests(unittest.TestCase):
    def test_known_idf_and_tie_order(self):
        index = BM25Index([
            passage("c", "оплата чек"),
            passage("b", "возврат деньги"),
            passage("a", "возврат оплата"),
        ])
        hits = index.search("возврат оплата")
        idf = math.log1p(1.5 / 2.5)
        self.assertEqual([hit.passage.document_id for hit in hits], ["a", "b", "c"])
        for hit, expected in zip(hits, [2 * idf, idf, idf]):
            self.assertAlmostEqual(hit.score, expected)

    def test_length_normalization_and_tf_saturation(self):
        documents = [passage("a", "срок"), passage("b", "срок общий ответ"),
                     passage("c", "срок срок срок")]
        scores = {hit.passage.document_id: hit.score for hit in BM25Index(documents).search("срок")}
        self.assertGreater(scores["a"], scores["b"])
        flat_scores = {hit.passage.document_id: hit.score for hit in BM25Index(documents, b=0).search("срок")}
        self.assertAlmostEqual(flat_scores["a"], flat_scores["b"])
        self.assertGreater(flat_scores["c"], flat_scores["a"])
        self.assertLess(flat_scores["c"], 3 * flat_scores["a"])

    def test_one_result_per_document_keeps_best_passage(self):
        index = BM25Index([
            passage("a", "перевод деньги"),
            passage("a", "перевод перевод", 2),
            passage("b", "перевод взнос"),
        ])
        hits = index.search("перевод", top_k=2)
        self.assertEqual([hit.passage.passage_id for hit in hits], ["a#02", "b#01"])
        candidates = index.search("перевод", top_k=3, passages_per_document=2)
        self.assertEqual([hit.passage.passage_id for hit in candidates], ["a#02", "a#01", "b#01"])

    def test_empty_unknown_and_repeated_query_terms(self):
        index = BM25Index([passage("a", "перевод деньги")])
        self.assertEqual(index.search("..."), [])
        self.assertEqual(index.search("А что и как?"), [])
        self.assertEqual(index.search("космодром"), [])
        self.assertEqual(index.search("перевод перевод"), index.search("перевод"))

    def test_normalizes_law_number_and_yo(self):
        tokenize = make_tokenizer("basic")
        self.assertEqual(tokenize("СЧЁТ № 442 — ФЗ, 1,4 млн"), ["счет", "442-фз", "1.4", "млн"])
        for law_number in ("115-ФЗ", "115 фз", "115ФЗ"):
            with self.subTest(law_number=law_number):
                self.assertEqual(tokenize(law_number), ["115-фз"])

    def test_finds_sources_for_checkpoint_topics(self):
        index = SearchIndex(load_corpus(CORPUS_PATH), tokenizer="stem")
        for question, expected in (
            ("банкротство", "bankruptcy-court"),
            ("115 фз", "aml-law-overview"),
            ("европротокол", "cbr-faq-insh-evroprotokol"),
            ("ипотечные каникулы", "cbr-faq-bank-s-ipotechnye-kanikuly"),
            ("цифровой рубль", "cbr-faq-dr"),
            ("показатель долговой нагрузки", "cbr-debt-burden"),
            ("Можно ли ИП быть самозанятым?", "fns-npd"),
            ("Совместимость УСН и НПД", "fns-npd-transition"),
            ("НПД и трудовой договор", "fns-npd-employment"),
            ("Регистрация индивидуального предпринимателя", "fns-ip-registration"),
            ("Прекращение деятельности индивидуального предпринимателя", "fns-ip-termination"),
        ):
            with self.subTest(question=question):
                hits = index.search(question, top_k=10, passages_per_document=3)
                self.assertIn(expected, [hit.passage.document_id for hit in hits])

    def test_rejects_invalid_search_parameters(self):
        documents = [passage("a", "перевод")]
        with self.assertRaises(ValueError):
            BM25Index(documents, k1=math.nan)
        with self.assertRaises(ValueError):
            BM25Index(documents, b=1.1)
        with self.assertRaises(ValueError):
            BM25Index(documents).search("перевод", top_k=0)


class MetricTests(unittest.TestCase):
    def test_recall_uses_all_gold_documents_and_rr_uses_first_hit(self):
        self.assertEqual(score_query(["x", "a", "b", "c"], {"a", "b", "c", "d"}, 3), (0.5, 0.5))

    def test_duplicate_documents_and_miss(self):
        self.assertEqual(score_query(["x", "x", "b"], {"a", "b", "c"}, 2), (1 / 3, 0.5))
        self.assertEqual(score_query(["x"], {"a", "b", "c"}, 3), (0, 0))


class AnswerTests(unittest.TestCase):
    def test_uses_coarena_limits_to_confirm_the_free_plan(self):
        limits = BytesIO(b'{"data":{"plan":"free"}}')
        limits.status = 200
        completion = BytesIO(b'{"model":"model","choices":[{"message":{"content":"Answer [1]."}}]}')
        completion.status = 200
        with patch("demo.answer.http.client.HTTPSConnection") as https:
            connection = https.return_value
            connection.getresponse.side_effect = [limits, completion]
            result = chat_completion([], api_key="test-key", model=COARENA_FREE_MODELS[0])

        self.assertEqual(result, ("Answer [1].", "model"))
        self.assertEqual([call.args[:2] for call in connection.request.call_args_list],
                         [("GET", "/api/v1/limits"), ("POST", "/api/v1/chat/completions")])
        payload = json.loads(connection.request.call_args.args[2])
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["max_tokens"], 1200)

    def test_archive_publication_can_be_found_by_its_requisites(self):
        reference = passage("faq", "Перевод денежных средств, Положение 762-П.")
        publication = replace(
            passage("publication", "Норма о предъявлении распоряжения."),
            title="№ 762-П от 2021-06-29 · О правилах осуществления перевода денежных средств",
            kind="archive_publication",
            section="Страница 1",
            updated="2021-08-30",
        )
        index = SearchIndex([reference, publication])
        named = index.search("Найди первоначальную публикацию 762-П", top_k=1)
        self.assertEqual(named[0].passage.document_id, "publication")
        self.assertEqual(index.search("Как осуществить перевод?")[0].passage.document_id, "faq")

    def test_direct_statutory_reference_keeps_the_named_provision_first(self):
        reference = passage("faq", "Что предусмотрено пунктом 13.5 статьи 7 закона 115-ФЗ?")
        law = replace(passage("law", "13.5. Основной текст нормы."),
                      title="Федеральный закон № 115-ФЗ", section="Статья 7. Права и обязанности",
                      kind="effective_law")
        distractor = replace(law, document_id="other", title="Закон № 116-ФЗ о применении 115-ФЗ")
        index = SearchIndex([reference, distractor, law])
        result = answer_question(index, "Что предусмотрено пунктом 13.5 статьи 7 закона 115-ФЗ?",
                                 lambda _: ("Ответ [1].", "model"))
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["sources"][0]["document_id"], "law")

    def test_reformulates_a_missing_match_and_returns_the_new_citation(self):
        index = BM25Index([passage("a", "идентификация паспорт")])
        calls = []

        def generate(messages):
            calls.append(messages)
            if len(calls) == 1:
                self.assertIn("нет результатов", messages[1]["content"])
                return "ПОИСК: идентификация", "model"
            self.assertIn("Поиск завершён", messages[0]["content"])
            self.assertIn("проверите личность", messages[1]["content"])
            self.assertIn("идентификация паспорт", messages[1]["content"])
            return "Нужен паспорт [1].", "model"

        result = answer_question(index, "Как вы проверите личность?", generate)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "answered")
        self.assertEqual([source["document_id"] for source in result["sources"]], ["a"])

    def test_review_can_refine_a_request_declined_in_fast_mode(self):
        index = BM25Index([passage("a", "клиент и банк"), passage("b", "идентификация паспорт")])

        def generate(messages):
            if "Поиск завершён" in messages[0]["content"]:
                return "Нужен паспорт [1].", "model"
            return f"{INSUFFICIENT} Нужен уточнённый поиск.", "model"

        result = answer_question(index, "Как банк проверит клиента?", generate,
                                 review=lambda _: (f"{INSUFFICIENT}\nПОИСК: идентификация", "model"))
        self.assertEqual(result["status"], "answered")
        self.assertEqual([source["document_id"] for source in result["sources"]], ["b"])

    def test_skips_a_second_completion_if_search_does_not_change_the_context(self):
        index = BM25Index([passage("a", "перевод деньги")])
        calls = []

        def generate(messages):
            calls.append(messages)
            return "ПОИСК: перевод деньги", "model"

        result = answer_question(index, "перевод", generate)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["answer"], INSUFFICIENT)

    def test_rejects_an_answer_with_an_unknown_reference(self):
        index = BM25Index([passage("a", "перевод деньги")])
        result = answer_question(index, "перевод", lambda _: ("Ответ [2].", "model"))
        self.assertEqual(result["status"], "generation_error")
        self.assertEqual([source["number"] for source in result["sources"]], [1])
