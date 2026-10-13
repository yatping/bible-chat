from bible_ft.split import leaked_questions, normalise_question, split_records


def rec(rid, book, question):
    return {"id": rid, "book": book, "inputs": {"question": question}}


def test_split_holds_out_books_and_has_no_leakage():
    records = [rec(f"r{i}", "Matthew" if i % 2 else "John", f"Question number {i}?") for i in range(40)]
    records += [rec("ruth1", "Ruth", "What did Ruth say?"), rec("ruth2", "Ruth", "What did Ruth say?")]
    # Same question asked about a non-held-out book must follow the test copy.
    records += [rec("leak", "General", "what did RUTH say")]
    records += [rec("dup-a", "Matthew", "Shared question"), rec("dup-b", "John", "Shared question")]
    splits = split_records(records, ["Ruth"], val_fraction=0.1, seed=1)

    assert {r["id"] for r in splits["test"]} == {"ruth1", "ruth2", "leak"}
    assert sum(len(v) for v in splits.values()) == len(records)
    assert leaked_questions(splits) == set()
    assert len(splits["val"]) >= round(0.1 * len(records))
    assert all(r["book"] != "Ruth" for r in splits["train"] + splits["val"])


def test_split_is_deterministic():
    records = [rec(f"r{i}", "Mark", f"q{i}") for i in range(30)]
    a = split_records(records, ["Ruth"], seed=5)
    b = split_records(records, ["Ruth"], seed=5)
    assert [r["id"] for r in a["val"]] == [r["id"] for r in b["val"]]


def test_normalise_question():
    assert normalise_question("  What did RUTH say?! ") == "what did ruth say"
