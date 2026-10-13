# Bundled content

`scripts/bundle_content.sh` copies the content pipeline output here (gitignored):

- `bible.sqlite` — read-only content DB (`shared/schema_content.sql`)
- `rag_embeddings.f16` — document embeddings (CONTRACTS §2)

The folder is added to the app as a folder reference, so the files end up in `BibleAI.app/Content/`.
If they are missing the app still builds and runs, and shows a "content not bundled" message.
