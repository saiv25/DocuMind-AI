import unittest

from langchain_core.documents import Document

from rag_engine import (
    clean_filename,
    safe_collection_name,
    stable_chunk_id,
)


class DocuMindHelperTests(unittest.TestCase):

    def test_collection_name_is_deterministic(self):
        files = [
            ("b.pdf", b"two"),
            ("a.pdf", b"one"),
        ]

        self.assertEqual(
            safe_collection_name(
                files,
                "alice",
                False,
                False,
                "vision",
            ),
            safe_collection_name(
                list(reversed(files)),
                "alice",
                False,
                False,
                "vision",
            ),
        )

    def test_collections_are_isolated_by_workspace(self):
        files = [
            ("a.pdf", b"document"),
        ]

        a = safe_collection_name(
            files,
            "alice",
            False,
            False,
            "vision",
        )

        b = safe_collection_name(
            files,
            "bob",
            False,
            False,
            "vision",
        )

        self.assertNotEqual(a, b)

    def test_collection_changes_with_ocr_setting(self):
        files = [
            ("a.pdf", b"document"),
        ]

        self.assertNotEqual(
            safe_collection_name(
                files,
                "alice",
                False,
                False,
                "vision",
            ),
            safe_collection_name(
                files,
                "alice",
                True,
                False,
                "vision",
            ),
        )

    def test_stable_chunk_id_is_content_sensitive(self):
        metadata = {
            "file_hash": "abc",
            "source": "test.pdf",
            "page": 1,
            "start_index": 0,
            "content_kind": "text",
        }

        first = Document(
            page_content="some text",
            metadata=metadata,
        )

        same = Document(
            page_content="some text",
            metadata=metadata,
        )

        changed = Document(
            page_content="different text",
            metadata=metadata,
        )

        self.assertEqual(
            stable_chunk_id(first),
            stable_chunk_id(same),
        )

        self.assertNotEqual(
            stable_chunk_id(first),
            stable_chunk_id(changed),
        )

    def test_filename_is_sanitised(self):
        self.assertNotIn(
            "/",
            clean_filename("../../unsafe/name.pdf"),
        )

        self.assertNotIn(
            "\\",
            clean_filename("..\\unsafe\\name.pdf"),
        )


if __name__ == "__main__":
    unittest.main()