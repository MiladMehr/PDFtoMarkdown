import json
import math
import os
import time
from google import genai
from pypdf import PdfReader, PdfWriter

# Clear the terminal screen before running
os.system("cls" if os.name == "nt" else "clear")

# Read API key from environment variable (do not hardcode secret keys in repository)
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Set it before running the script (e.g., export GEMINI_API_KEY='your_api_key')."
    )

# 1. Initialize the Google GenAI client
client = genai.Client(api_key=API_KEY)

# Folder containing PDF files (relative to the current working directory)
INPUT_FOLDER = "Your_PDFs"

# Model name used for conversion
MODEL_NAME = "gemini-3.8-flash"

# Chunking thresholds for large PDFs
MAX_PAGES_PER_CHUNK = 10       # Maximum pages per chunk
MAX_SIZE_MB_PER_CHUNK = 12.0   # Maximum approximate size per chunk (MB)

# Retry settings for transient errors
MAX_RETRIES = 4
BASE_WAIT_SECONDS = 5

# Conversion prompt placed after input and runtime configuration
prompt = (
    "Please convert this entire document into clean and accurate Markdown format. "
    "Preserve all mathematical formulas, tables, and structures correctly, "
    "and do not output any control characters or NUL bytes."
)


def is_retryable_error(err: Exception) -> bool:
    msg = str(err).upper()
    return any(code in msg for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL"))


def get_pdf_files_in_folder(folder_path: str) -> list[str]:
    if not os.path.isdir(folder_path):
        return []
    files = []
    for entry in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path) and entry.lower().endswith(".pdf"):
            if not entry.startswith("(GeminiMarkdown)"):
                files.append(full_path)
    return files


def build_part_plan(pdf_path: str, parts_dir: str) -> tuple[int, list[dict]]:
    """
    Creates or validates the chunk plan for a PDF and generates missing PDF part files.
    Each item in plan: {"index": int, "start_page": int, "end_page": int, "pdf_part": str, "md_part": str}
    """
    os.makedirs(parts_dir, exist_ok=True)
    manifest_path = os.path.join(parts_dir, "manifest.json")

    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    if total_pages == 0:
        raise ValueError("PDF has 0 pages.")

    needs_split = (total_pages > MAX_PAGES_PER_CHUNK) or (file_size_mb > MAX_SIZE_MB_PER_CHUNK)
    if not needs_split:
        num_chunks = 1
        pages_per_part = total_pages
    else:
        chunks_by_pages = math.ceil(total_pages / MAX_PAGES_PER_CHUNK)
        chunks_by_size = math.ceil(file_size_mb / MAX_SIZE_MB_PER_CHUNK)
        num_chunks = max(chunks_by_pages, chunks_by_size)
        pages_per_part = math.ceil(total_pages / num_chunks)

    expected_ranges = []
    for part_idx in range(num_chunks):
        start_idx = part_idx * pages_per_part
        end_idx = min(start_idx + pages_per_part, total_pages)
        if start_idx >= total_pages:
            break
        expected_ranges.append((part_idx + 1, start_idx + 1, end_idx))

    actual_num_chunks = len(expected_ranges)

    # Validate existing manifest; rebuild part files if page count or chunk structure changed
    manifest_valid = False
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f_m:
                old_manifest = json.load(f_m)
            if (
                old_manifest.get("total_pages") == total_pages
                and old_manifest.get("num_chunks") == actual_num_chunks
                and old_manifest.get("ranges") == [list(r) for r in expected_ranges]
            ):
                manifest_valid = True
        except Exception:
            manifest_valid = False

    if not manifest_valid:
        for old_entry in os.listdir(parts_dir):
            if old_entry != "manifest.json":
                old_full = os.path.join(parts_dir, old_entry)
                if os.path.isfile(old_full):
                    os.remove(old_full)

    plan = []
    for part_num, start_p, end_p in expected_ranges:
        part_pdf_path = os.path.join(parts_dir, f"part_{part_num:03d}.pdf")
        part_md_path = os.path.join(parts_dir, f"part_{part_num:03d}.md")

        if actual_num_chunks == 1:
            source_pdf_for_part = pdf_path
        else:
            if not os.path.exists(part_pdf_path) or os.path.getsize(part_pdf_path) == 0:
                writer = PdfWriter()
                for page_num in range(start_p - 1, end_p):
                    writer.add_page(reader.pages[page_num])
                with open(part_pdf_path, "wb") as f_out:
                    writer.write(f_out)
            source_pdf_for_part = part_pdf_path

        plan.append({
            "index": part_num,
            "start_page": start_p,
            "end_page": end_p,
            "pdf_part": source_pdf_for_part,
            "md_part": part_md_path,
        })

    with open(manifest_path, "w", encoding="utf-8") as f_m:
        json.dump(
            {
                "total_pages": total_pages,
                "num_chunks": actual_num_chunks,
                "ranges": [list(r) for r in expected_ranges],
            },
            f_m,
            ensure_ascii=False,
            indent=2,
        )

    return actual_num_chunks, plan


def convert_single_pdf(pdf_path: str, label_for_log: str) -> tuple[bool, str, int]:
    """
    Converts a single PDF file with retries and remote file cleanup.
    Returns: (success, markdown_text, attempts_used)
    """
    attempts_used = 0

    for attempt in range(1, MAX_RETRIES + 1):
        attempts_used = attempt
        uploaded_file = None
        try:
            uploaded_file = client.files.upload(file=pdf_path)
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[uploaded_file, prompt],
            )
            return True, (response.text or ""), attempts_used

        except Exception as e:
            if attempt < MAX_RETRIES and is_retryable_error(e):
                wait_time = BASE_WAIT_SECONDS * attempt
                print(
                    f"  Temporary error on {label_for_log} (attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                print(f"  Failed on {label_for_log} (attempt {attempt}/{MAX_RETRIES}): {e}")
                break

        finally:
            if uploaded_file is not None:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception as cleanup_err:
                    print(f"  Warning: could not delete remote file '{uploaded_file.name}': {cleanup_err}")

    return False, "", attempts_used


def merge_parts_to_final(num_parts: int, plan: list[dict], output_file: str) -> None:
    if num_parts == 1:
        with open(plan[0]["md_part"], "r", encoding="utf-8") as f_single:
            content = f_single.read()
        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.write(content)
        return

    merged_sections = []
    for item in plan:
        with open(item["md_part"], "r", encoding="utf-8") as f_part:
            section_text = f_part.read().strip()
        merged_sections.append(
            f"<!-- Part {item['index']}/{num_parts}: Pages {item['start_page']}-{item['end_page']} -->\n\n{section_text}"
        )

    final_markdown = "\n\n---\n\n".join(merged_sections) + "\n"
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(final_markdown)


results = []

# 2. Scan input folder and process PDF files
if not os.path.isdir(INPUT_FOLDER):
    print(f"Input folder '{INPUT_FOLDER}' not found in current directory: {os.getcwd()}")
else:
    pdf_files = get_pdf_files_in_folder(INPUT_FOLDER)
    print(f"Found {len(pdf_files)} PDF file(s) in '{INPUT_FOLDER}'.")

    for idx, input_file in enumerate(pdf_files):
        # Two-line separation between files in terminal output
        if idx > 0:
            print("\n")

        base_name = os.path.basename(input_file)
        file_name_without_ext, _ = os.path.splitext(base_name)
        output_file = os.path.join(INPUT_FOLDER, f"(GeminiMarkdown) {file_name_without_ext}.md")
        output_name = os.path.basename(output_file)
        parts_dir = os.path.join(INPUT_FOLDER, f"_md_parts_{file_name_without_ext}")

        print(f"Processing '{base_name}'...")

        # Skip if the final Markdown file already exists and is non-empty
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"  Final output already exists: '{output_file}'. Skipping.")
            results.append({
                "file": base_name,
                "status": "Already Exists",
                "parts_ok": "-",
                "attempts": 0,
                "output": output_name,
            })
            continue

        total_attempts_for_file = 0

        try:
            num_parts, plan = build_part_plan(input_file, parts_dir)
            if num_parts > 1:
                print(f"  Document has {num_parts} part(s). Checking existing and missing Markdown parts...")

            all_parts_ready = True
            succeeded_parts = 0

            for item in plan:
                part_label = f"part {item['index']}/{num_parts} [pages {item['start_page']}-{item['end_page']}]"
                md_part_path = item["md_part"]

                # Reuse already converted non-empty part Markdown files
                if os.path.exists(md_part_path) and os.path.getsize(md_part_path) > 0:
                    succeeded_parts += 1
                    print(f"  Existing {part_label} Markdown found. Keeping it.")
                    continue

                print(f"  Converting {part_label}...")
                ok, md_text, tries = convert_single_pdf(item["pdf_part"], f"'{base_name}' {part_label}")
                total_attempts_for_file += tries

                if ok:
                    with open(md_part_path, "w", encoding="utf-8") as f_part:
                        f_part.write(md_text)
                    succeeded_parts += 1
                    print(f"  Saved {part_label} Markdown (attempt {tries}).")
                else:
                    all_parts_ready = False
                    print(f"  ERROR: {part_label} failed after {tries} attempt(s).")
                    break

            if all_parts_ready:
                merge_parts_to_final(num_parts, plan, output_file)
                print(f"  Saved final Markdown to '{output_file}'.")
                results.append({
                    "file": base_name,
                    "status": "Success",
                    "parts_ok": f"{succeeded_parts}/{num_parts}",
                    "attempts": total_attempts_for_file,
                    "output": output_name,
                })
            else:
                print(
                    f"  Incomplete '{base_name}' ({succeeded_parts}/{num_parts} parts ready). "
                    f"Completed parts are saved in '{parts_dir}'; run again later to finish missing parts."
                )
                results.append({
                    "file": base_name,
                    "status": f"Incomplete ({succeeded_parts}/{num_parts})",
                    "parts_ok": f"{succeeded_parts}/{num_parts}",
                    "attempts": total_attempts_for_file,
                    "output": "-",
                })

        except Exception as e:
            print(f"  Unexpected error on '{base_name}': {e}")
            results.append({
                "file": base_name,
                "status": "Failed",
                "parts_ok": "-",
                "attempts": total_attempts_for_file,
                "output": "-",
            })

# 3. Print aligned final summary table
success_count = sum(1 for r in results if r["status"] == "Success")
exists_count = sum(1 for r in results if r["status"] == "Already Exists")
incomplete_count = sum(1 for r in results if r["status"].startswith("Incomplete"))
failed_count = sum(1 for r in results if r["status"] == "Failed")
total_attempts = sum(r["attempts"] for r in results)

file_col_w = max([len("File")] + [len(r["file"]) for r in results])
status_col_w = max([len("Status")] + [len(r["status"]) for r in results])
parts_col_w = max([len("Parts")] + [len(r["parts_ok"]) for r in results])
tries_col_w = max([len("Tries")] + [len(str(r["attempts"])) for r in results])
out_col_w = max([len("Output")] + [len(r["output"]) for r in results])

header_line = (
    f"{'File'.ljust(file_col_w)} | "
    f"{'Status'.ljust(status_col_w)} | "
    f"{'Parts'.ljust(parts_col_w)} | "
    f"{'Tries'.rjust(tries_col_w)} | "
    f"{'Output'.ljust(out_col_w)}"
)
table_width = len(header_line)

print("\n\n" + "=" * table_width)
print("SUMMARY")
print("=" * table_width)
print(f"Total PDFs     : {len(results)}")
print(f"Succeeded      : {success_count}")
print(f"Already Exists : {exists_count}")
print(f"Incomplete     : {incomplete_count}")
print(f"Failed         : {failed_count}")
print(f"Total tries    : {total_attempts}")
print("-" * table_width)
print(header_line)
print("-" * table_width)

for r in results:
    row_line = (
        f"{r['file'].ljust(file_col_w)} | "
        f"{r['status'].ljust(status_col_w)} | "
        f"{r['parts_ok'].ljust(parts_col_w)} | "
        f"{str(r['attempts']).rjust(tries_col_w)} | "
        f"{r['output'].ljust(out_col_w)}"
    )
    print(row_line)

print("=" * table_width)