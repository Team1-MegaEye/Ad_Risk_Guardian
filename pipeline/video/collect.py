# -*- coding: utf-8 -*-
"""
Data collection script for deepfake detection project.

1) Split FaceForensics++ (FFPP) into train/test with approximate 1:1
   balance for (gender, real/fake), targeting TOTAL_FILES videos.
2) Split KoDF into train/test with approximate 1:1 balance for
   (gender, original/fake), targeting TOTAL videos.
3) Extract a fixed number of frames per video using a consistent
   time-based sampling rule based on the most frequent FPS (base_fps),
   and save frame paths into an Excel manifest.

Author: (Your name)
"""

import os
import re
import csv
import random
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
import pandas as pd
import shutil

# ---------------------------------------------------------
# Common settings
# ---------------------------------------------------------
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv"}


# =========================================================
# PART 0. Common utilities
# =========================================================
def reset_output_root(root: Path) -> None:
    """
    Remove an existing directory and recreate it empty.

    Parameters
    ----------
    root : Path
        Target directory to reset.
    """
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def ensure_dir(p: Path) -> None:
    """
    Create directory p (and parents) if it does not exist.
    """
    p.mkdir(parents=True, exist_ok=True)


def get_id_from_name(path: Path) -> str:
    """
    Extract 'ID' from filename.

    Examples
    --------
    003.mp4     -> '003'
    003_001.mp4 -> '003'
    """
    stem = path.stem
    return stem.split("_")[0]


def iter_videos(root: Path):
    """
    Yield all video files under a given root directory.

    Parameters
    ----------
    root : Path
        Root directory to search for videos.

    Yields
    ------
    Path
        Path to each video file.
    """
    if not root.exists():
        return
    for p in root.rglob("*.*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def make_unique(dst: Path) -> Path:
    """
    If dst already exists, append _dup{n} to make a unique filename.

    Parameters
    ----------
    dst : Path
        Desired file path.

    Returns
    -------
    Path
        Unique file path that does not exist yet.
    """
    if not dst.exists():
        return dst
    base, suf = dst.stem, dst.suffix
    for i in range(1, 10000):
        cand = dst.with_name(f"{base}_dup{i}{suf}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Too many duplicates for {dst}")


# =========================================================
# PART 1. FaceForensics++(C23) split (target total ≈ 270)
# =========================================================

CONFIG_FFPP = {
    # gender 폴더 (Man / Woman 하위에 ID별 mp4)
    "gender_root": Path(r"D:\deoha\Documents\video_proj\Dataset\FaceForensics++_C23\gender"),  # 입력 경로 설정 

    # DeepFake_Root (original + 변조 5종)
    "deepfake_root": Path(r"D:\deoha\Documents\video_proj\Dataset\FaceForensics++_C23\DeepFake_Root"),  # 입력 경로 설정
    "original_dirname": "original",
    "fake_dirnames": ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"],

    # 출력 폴더
    "output_root": Path(r"D:\deoha\Documents\video_proj\Dataset\FaceForensics++_C23\split_300_8_2"),  # 출력 경로 설정 

    # Target total number of video files (train + test, real + fake)
    "total_files": 270,   # each ID → (real, fake) = 2 files

    # train:test ratio
    "train_ratio": 0.8,

    "seed": 42,
}

random.seed(CONFIG_FFPP["seed"])


def scan_fake_candidates_ffpp(fake_roots, exts):
    """
    Build a mapping from ID to candidate fake videos and collect all fake paths.

    Parameters
    ----------
    fake_roots : list[Path]
        List of deepfake variant roots.
    exts : set[str]
        Allowed video extensions.

    Returns
    -------
    dict
        Mapping from id(str) to list of fake video paths.
    list[Path]
        Flat list of all fake video paths.
    """
    id_to_fakes = defaultdict(list)
    all_fakes = []

    for root in fake_roots:
        if not root.exists():
            print(f"[FFPP][WARN] fake root not found: {root}")
            continue
        for p in iter_videos(root):
            vid_id = get_id_from_name(p)
            id_to_fakes[vid_id].append(p)
            all_fakes.append(p)

    print(f"[FFPP] fake 후보 ID 개수: {len(id_to_fakes)}")
    print(f"[FFPP] fake 전체 파일 수: {len(all_fakes)}")
    return id_to_fakes, all_fakes


def pick_fake_for_id(id_to_fakes, vid_id):
    """
    Randomly pick one fake candidate for a given ID.
    If no candidate exists, return None.
    """
    cands = id_to_fakes.get(vid_id, [])
    if not cands:
        return None
    return random.choice(cands)


def split_faceforensics() -> None:
    """
    Split FaceForensics++ into train/test with approximate gender balance
    and exact total_files (if enough IDs are available).

    - Target: CONFIG_FFPP["total_files"] videos in total.
      Each ID contributes (real, fake) = 2 videos.
    - So number of IDs used = total_files // 2.
    - Gender balance: number of male/female IDs differs by at most 1.
    """
    cfg = CONFIG_FFPP
    gender_root = cfg["gender_root"]
    deepfake_root = cfg["deepfake_root"]
    original_root = deepfake_root / cfg["original_dirname"]
    fake_roots = [deepfake_root / d for d in cfg["fake_dirnames"]]

    out_root = cfg["output_root"]
    exts = VIDEO_EXTS

    reset_output_root(out_root)

    # 1) Scan fake candidates
    id_to_fakes, all_fakes = scan_fake_candidates_ffpp(fake_roots, exts)

    # 2) Collect male/female video files from gender folders
    man_dir = gender_root / "Man"
    woman_dir = gender_root / "Woman"

    def collect_files(gender_dir):
        return [p for p in iter_videos(gender_dir)]

    man_files = collect_files(man_dir)
    woman_files = collect_files(woman_dir)

    print(f"[FFPP] Man 폴더 video 파일 수: {len(man_files)}")
    print(f"[FFPP] Woman 폴더 video 파일 수: {len(woman_files)}")

    man_id_to_path = {}
    woman_id_to_path = {}

    for p in man_files:
        vid_id = get_id_from_name(p)
        man_id_to_path[vid_id] = p

    for p in woman_files:
        vid_id = get_id_from_name(p)
        woman_id_to_path[vid_id] = p

    num_male_ids = len(man_id_to_path)
    num_female_ids = len(woman_id_to_path)

    print(f"[FFPP] 남자 ID 후보 수 (전체): {num_male_ids}")
    print(f"[FFPP] 여자 ID 후보 수 (전체): {num_female_ids}")

    # 3) Decide how many IDs to use (approx. balanced M/F, exact total_files if possible)
    total_files_target = cfg["total_files"]
    total_ids_target = total_files_target // 2  # each ID -> 2 files (real + fake)

    max_ids_available = num_male_ids + num_female_ids
    if max_ids_available < total_ids_target:
        # Not enough IDs overall -> reduce target, keep balance as much as possible
        print(
            f"[FFPP][WARN] Not enough IDs to reach {total_files_target} files.\n"
            f"  Available IDs: {max_ids_available} → "
            f"max possible files: {max_ids_available * 2}"
        )
        total_ids_target = max_ids_available

    # Base split: approximately half male, half female
    base_each = total_ids_target // 2
    remainder = total_ids_target % 2  # if odd, give +1 ID to one gender

    # First proposal: male gets base_each + (1 if remainder else 0)
    target_male_ids = min(num_male_ids, base_each + (1 if remainder > 0 else 0))
    target_female_ids = min(num_female_ids, total_ids_target - target_male_ids)

    # If we still haven't reached total_ids_target, try to allocate remaining IDs
    # to the gender that has available capacity.
    current_total_ids = target_male_ids + target_female_ids
    if current_total_ids < total_ids_target:
        remaining = total_ids_target - current_total_ids
        for _ in range(remaining):
            male_capacity_left = num_male_ids - target_male_ids
            female_capacity_left = num_female_ids - target_female_ids

            if male_capacity_left <= 0 and female_capacity_left <= 0:
                break  # no more IDs to allocate

            # Prefer the gender with more remaining capacity
            if male_capacity_left >= female_capacity_left and male_capacity_left > 0:
                target_male_ids += 1
            elif female_capacity_left > 0:
                target_female_ids += 1

    final_ids_used = target_male_ids + target_female_ids
    final_files = final_ids_used * 2

    print(
        f"[FFPP] 최종 ID 사용 수: 남 {target_male_ids} / 여 {target_female_ids} "
        f"(총 ID={final_ids_used}, 총 영상={final_files})"
    )

    if final_ids_used == 0:
        print("[FFPP][ERROR] 사용할 수 있는 ID가 없습니다. FFPP split 생략.")
        return

    # 4) Sample IDs by gender
    all_male_ids = list(man_id_to_path.keys())
    all_female_ids = list(woman_id_to_path.keys())

    random.shuffle(all_male_ids)
    random.shuffle(all_female_ids)

    man_ids = all_male_ids[:target_male_ids]
    woman_ids = all_female_ids[:target_female_ids]

    # 5) Split each gender into train/test
    train_ratio = cfg["train_ratio"]

    man_train_count = int(target_male_ids * train_ratio)
    woman_train_count = int(target_female_ids * train_ratio)

    man_train_ids = man_ids[:man_train_count]
    man_test_ids = man_ids[man_train_count:]

    woman_train_ids = woman_ids[:woman_train_count]
    woman_test_ids = woman_ids[woman_train_count:]

    print(f"[FFPP] 남자 train/test: {len(man_train_ids)} / {len(man_test_ids)}")
    print(f"[FFPP] 여자 train/test: {len(woman_train_ids)} / {len(woman_test_ids)}")

    original_root_exists = original_root.exists()
    if not original_root_exists:
        print(f"[FFPP][WARN] original_root가 존재하지 않습니다: {original_root}")
        print("           → real도 gender 폴더 파일 그대로 사용합니다.")

    copied_count = 0

    def copy_pair(split_name: str, gender: str, vid_id: str) -> None:
        """
        Copy a (real, fake) pair for the given ID into the split folder.

        Parameters
        ----------
        split_name : {"train", "test"}
        gender : {"Man", "Woman"}
        vid_id : str
        """
        nonlocal copied_count

        if gender == "Man":
            gender_src = man_id_to_path[vid_id]
        else:
            gender_src = woman_id_to_path[vid_id]

        # Prefer original_root if available, fall back to gender folder
        candidate_real = original_root / gender_src.name if original_root_exists else None
        real_src = candidate_real if candidate_real and candidate_real.exists() else gender_src

        # Pick one fake for this ID, fallback to global fake pool if none
        fake_src = pick_fake_for_id(id_to_fakes, vid_id)
        if fake_src is None:
            fake_src = random.choice(all_fakes)
            print(f"[FFPP][INFO] ID={vid_id}에 해당하는 fake 없음 → 랜덤 fake 사용: {fake_src.name}")

        real_dst_dir = out_root / split_name / "real"
        fake_dst_dir = out_root / split_name / "fake"
        ensure_dir(real_dst_dir)
        ensure_dir(fake_dst_dir)

        real_dst = make_unique(real_dst_dir / real_src.name)
        fake_dst = make_unique(fake_dst_dir / fake_src.name)

        shutil.copy2(real_src, real_dst)
        shutil.copy2(fake_src, fake_dst)
        copied_count += 2  # real + fake

    # Copy train/test pairs
    for vid_id in man_train_ids:
        copy_pair("train", "Man", vid_id)
    for vid_id in woman_train_ids:
        copy_pair("train", "Woman", vid_id)

    for vid_id in man_test_ids:
        copy_pair("test", "Man", vid_id)
    for vid_id in woman_test_ids:
        copy_pair("test", "Woman", vid_id)

    print(f"[FFPP] split 완료. 복사된 영상 수(real+fake): {copied_count} 개")


# =========================================================
# PART 2. KoDF split (target total ≈ 630)
# =========================================================

CONFIG_KODF = {
    # 원본 폴더
    "original_root": Path(r"D:\deoha\Documents\3_kind_Dataset\딥페이크 변조 영상\1.Training\[원천][원본]원본1"),  #  입력 경로 설정

    # 변조 폴더들 (여러 개)
    "others_roots": [
        Path(r"D:\deoha\Documents\3_kind_Dataset\딥페이크 변조 영상\1.Training\[원천][변조]train_dffs1_data"),  #  입력 경로 설정
        Path(r"D:\deoha\Documents\3_kind_Dataset\딥페이크 변조 영상\1.Training\[원천][변조]train_dfl1_data"),   #  입력 경로 설정
        Path(r"D:\deoha\Documents\3_kind_Dataset\딥페이크 변조 영상\1.Training\[원천][변조]train_fo1_data"),    #  입력 경로 설정
        Path(r"D:\deoha\Documents\3_kind_Dataset\딥페이크 변조 영상\1.Training\[원천][변조]train_fsgan1_data"), #  입력 경로 설정
        Path(r"D:\deoha\Documents\3_kind_Dataset\딥페이크 변조 영상\1.Training\[원천][변조]audio_driven1"),     #  입력 경로 설정
    ],

    # 메타데이터 (원본/변조)
    "metadata_csv_original": Path(
        r"D:\deoha\Documents\3_kind_Dataset\딥페이크 변조 영상\1.Training\[라벨링]train_meta\train_meta_data\원본영상_training_메타데이터.csv"
    ),
    "metadata_csv_fake": Path(
        r"D:\deoha\Documents\3_kind_Dataset\딥페이크 변조 영상\1.Training\[라벨링]train_meta\train_meta_data\변조영상_training_메타데이터.csv"
    ),  

    # KoDF 출력 폴더 (영상만 복사)
    "output_root": Path(
        r"D:\deoha\Documents\3_kind_Dataset\딥페이크 변조 영상\split_kodf_700_8_2"
    ),  # 출력 경로 설정

    # Target total number of videos (original + fake)
    "total": 630,
    "train_ratio": 0.8,
    "seed": 42,
}

UUID_RE = re.compile(
    r"(?i)([a-f0-9]{8}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{12})"
)


def extract_uuid_from_path(p: Path) -> str | None:
    """
    Extract UUID (hyphenated or not) from a path component and normalize it.

    Returns
    -------
    str or None
        Normalized UUID string (lowercase, no hyphens) or None if not found.
    """
    for part in [*p.parts[::-1]]:
        m = UUID_RE.search(part)
        if m:
            return m.group(1).replace("-", "").lower()
    return None


def _read_table(path: Path) -> pd.DataFrame:
    """
    Read CSV/Excel metadata file with basic encoding fallbacks.
    """
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"CSV 인코딩 판독 실패: {path}")


def _pick_col(df: pd.DataFrame, candidates, must: bool = False) -> str | None:
    """
    Pick one column from a list of candidate names (case-insensitive, partial match).
    """
    cols = list(df.columns)
    lowmap = {c.lower(): c for c in cols}
    # exact match
    for cand in candidates:
        if cand.lower() in lowmap:
            return lowmap[cand.lower()]
    # partial match
    for c in cols:
        cl = c.lower().replace(" ", "")
        for cand in candidates:
            if cand.lower().replace(" ", "") in cl:
                return c
    if must:
        raise ValueError(f"컬럼 미발견: 후보={candidates} / 실제={cols}")
    return None


def _norm_gender(g) -> str | None:
    """
    Normalize gender labels to 'M' or 'F'.
    """
    g = str(g).strip()
    if g in ("남성", "남", "남자", "M", "m"):
        return "M"
    if g in ("여성", "여", "여자", "F", "f"):
        return "F"
    return None


def _gender_by_videoid(vid_gender: dict, vp: Path) -> str | None:
    """
    Infer gender by matching video ID / filename recorded in metadata.
    """
    stem, name = vp.stem, vp.name
    g = vid_gender.get(stem) or vid_gender.get(name)
    if g:
        return g
    for k, gv in vid_gender.items():
        if len(k) >= 5 and (k.lower() in name.lower() or k.lower() in stem.lower()):
            return gv
    return None


def debug_scan_root(root: Path, label: str) -> None:
    """
    Simple debug helper to inspect a KoDF root directory.
    """
    print(f"[KODF][DEBUG] scan {label}: {root}")
    print("  exists:", root.exists(), " is_dir:", root.is_dir())
    if not root.exists():
        return
    any_files = list(root.rglob("*.*"))
    print("  any files count:", len(any_files))
    exts = Counter([p.suffix.lower() for p in any_files if p.is_file()])
    print("  top extensions:", exts.most_common(10))
    vids = [p for p in any_files if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    print("  video files count:", len(vids))
    for p in vids[:5]:
        print("   [video] ", p)


# ---------- Metadata loaders (original / fake) ----------

def build_gender_map_original(path: Path):
    """
    Build (UUID -> gender) and (videoID -> gender) maps from KoDF original metadata.
    """
    df = _read_table(path)
    print(f"[META-ORIG] rows={len(df)}  columns={list(df.columns)}")

    uuid_col = _pick_col(df, ["UUID", "uuid"])
    gender_col = _pick_col(df, ["인물성별", "성별", "gender"], must=True)
    videoid_col = _pick_col(df, ["영상ID", "영상_id", "영상id", "파일명", "파일이름", "영상파일명"])

    uuid_gender, vid_gender = {}, {}

    for _, row in df.iterrows():
        g = _norm_gender(row[gender_col])
        if g is None:
            continue
        if uuid_col is not None and pd.notna(row.get(uuid_col, None)):
            uuid_raw = str(row[uuid_col]).strip().lower()
            m = UUID_RE.fullmatch(uuid_raw) or UUID_RE.search(uuid_raw)
            if m:
                uuid_gender[m.group(1).replace("-", "").lower()] = g
        if videoid_col is not None and pd.notna(row.get(videoid_col, None)):
            vid_raw = str(row[videoid_col]).strip()
            vid_gender[Path(vid_raw).stem] = g
            vid_gender[Path(vid_raw).name] = g

    print(f"[META-ORIG] uuid_gender={len(uuid_gender)}  vid_gender={len(vid_gender)}")
    return uuid_gender, vid_gender


def build_gender_map_fake(path: Path):
    """
    Build (UUID -> gender) and (videoID -> gender) maps from KoDF fake metadata.
    """
    df = _read_table(path)
    print(f"[META-FAKE] rows={len(df)}  columns={list(df.columns)}")

    gender_col = _pick_col(df, ["인물성별", "성별", "gender"], must=True)
    tgt_uuid_col = _pick_col(df, ["타깃UUID", "타겟uuid", "targetuuid", "uuid"])
    src_uuid_col = _pick_col(df, ["소스UUID", "sourceuuid"])
    videoid_col = _pick_col(df, ["영상ID", "파일명", "파일이름", "영상파일명"])
    targetvid_col = _pick_col(df, ["타깃영상", "타겟영상", "targetvideo"])

    uuid_gender, vid_gender = {}, {}

    for _, row in df.iterrows():
        g = _norm_gender(row[gender_col])
        if g is None:
            continue
        for col in (tgt_uuid_col, src_uuid_col):
            if col and pd.notna(row.get(col, None)):
                uuid_raw = str(row[col]).strip().lower()
                m = UUID_RE.fullmatch(uuid_raw) or UUID_RE.search(uuid_raw)
                if m:
                    uuid_gender[m.group(1).replace("-", "").lower()] = g
        for col in (videoid_col, targetvid_col):
            if col and pd.notna(row.get(col, None)):
                vid_raw = str(row[col]).strip()
                vid_gender[Path(vid_raw).stem] = g
                vid_gender[Path(vid_raw).name] = g

    print(f"[META-FAKE] uuid_gender={len(uuid_gender)}  vid_gender={len(vid_gender)}")
    return uuid_gender, vid_gender


def collect_candidates_kodf(original_root: Path,
                            others_roots: list[Path],
                            orig_uuid: dict, orig_vid: dict,
                            fake_uuid: dict, fake_vid: dict):
    """
    Collect KoDF video samples with known gender information.

    Returns
    -------
    list[dict]
        Each dict has keys: {"path", "cls", "uuid", "gender"}.
        cls ∈ {"original", "fake"}.
    """
    samples = []

    def add_group(root: Path, cls_label: str):
        for vp in iter_videos(root):
            uuid = extract_uuid_from_path(vp)
            gender = None
            if cls_label == "original":
                if uuid:
                    gender = orig_uuid.get(uuid)
                if gender is None:
                    gender = _gender_by_videoid(orig_vid, vp)
            else:
                if uuid:
                    gender = fake_uuid.get(uuid)
                if gender is None:
                    gender = _gender_by_videoid(fake_vid, vp)

            samples.append({
                "path": str(vp),
                "cls": cls_label,
                "uuid": uuid,
                "gender": gender
            })

    add_group(original_root, "original")
    for r in others_roots:
        add_group(r, "fake")

    samples = [s for s in samples if s["gender"] in ("M", "F")]
    print(f"[KODF][SCAN] matched (gender known): {len(samples)}")
    return samples


def stratified_sample_kodf(samples, total: int):
    """
    Stratified sampling over 4 buckets: (original/fake) × (M/F).

    Goal:
    -----
    - Total number of picked samples ≈ total (exact if enough data).
    - Bucket-wise counts as balanced as possible.
      (difference between buckets is at most 1 when capacity allows)

    Parameters
    ----------
    samples : list[dict]
        KoDF candidate samples.
    total : int
        Target total number of videos to sample.

    Returns
    -------
    list[dict]
        Picked samples with approximate balance.
    """
    from collections import defaultdict

    # Build buckets
    buckets = {
        ("original", "M"): [],
        ("original", "F"): [],
        ("fake", "M"): [],
        ("fake", "F"): [],
    }
    for s in samples:
        k = (s["cls"], s["gender"])
        if k in buckets:
            buckets[k].append(s)

    print("[KODF] 버킷 현황:")
    total_available = 0
    for k, arr in buckets.items():
        print(f"  {k}: {len(arr)} 개")
        total_available += len(arr)

    # Adjust target if not enough samples overall
    target_total = total
    if total_available < total:
        print(
            f"[KODF][WARN] Requested total={total}, "
            f"but only {total_available} matched samples are available."
        )
        target_total = total_available

    # Initial ideal distribution: floor(total/4) per bucket, +1 to some buckets if remainder > 0
    bucket_keys = list(buckets.keys())
    per_bucket_floor = target_total // 4
    remainder = target_total % 4

    desired = {k: per_bucket_floor for k in bucket_keys}
    for i in range(remainder):
        desired[bucket_keys[i]] += 1  # distribute remainder

    # Respect capacity: if desired > capacity, reduce and track deficit
    deficit = 0
    capacity = {k: len(buckets[k]) for k in bucket_keys}

    for k in bucket_keys:
        if desired[k] > capacity[k]:
            deficit += desired[k] - capacity[k]
            desired[k] = capacity[k]

    # Try to redistribute deficit to buckets with remaining capacity
    if deficit > 0:
        extra_cap = {k: capacity[k] - desired[k] for k in bucket_keys}
        while deficit > 0:
            allocated = False
            for k in bucket_keys:
                if extra_cap[k] > 0 and deficit > 0:
                    desired[k] += 1
                    extra_cap[k] -= 1
                    deficit -= 1
                    allocated = True
                    if deficit == 0:
                        break
            if not allocated:
                # No more extra capacity anywhere
                break

    # Final sanity check: actual achievable total
    final_total = sum(desired[k] for k in bucket_keys)
    print(f"[KODF] 최종 버킷별 샘플 수(desired):")
    for k in bucket_keys:
        print(f"  {k}: {desired[k]} 개 (capacity={capacity[k]})")
    print(f"[KODF] 최종 target_total={target_total}, 실제 achievable={final_total}")

    picked = []
    for k in bucket_keys:
        arr = buckets[k]
        random.shuffle(arr)
        picked.extend(arr[:desired[k]])

    # If for some reason we still have fewer than target_total and capacity remains,
    # we can fill the gap with random remaining samples ignoring bucket balance.
    if final_total < target_total:
        print(f"[KODF][INFO] Filling remaining {target_total - final_total} samples with random picks.")
        used_ids = set(id(s) for s in picked)
        remaining = [s for s in samples if id(s) not in used_ids]
        random.shuffle(remaining)
        need = target_total - final_total
        picked.extend(remaining[:need])

    print(f"[KODF] 최종 picked 수: {len(picked)}")
    return picked


def stratified_split_kodf(picked, train_ratio: float = 0.8):
    """
    Split picked KoDF samples into train/test while preserving bucket ratio.
    """
    groups = {}
    for s in picked:
        groups.setdefault((s["cls"], s["gender"]), []).append(s)
    train, test = [], []
    for _, arr in groups.items():
        random.shuffle(arr)
        n_tr = int(round(len(arr) * train_ratio))
        train.extend(arr[:n_tr])
        test.extend(arr[n_tr:])
    return train, test


def save_manifest_kodf(rows, out_csv: Path) -> None:
    """
    Save KoDF split manifest as CSV.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "gender", "uuid", "video_path", "dest_path"])
        for r in rows:
            w.writerow([
                r["split"], r["cls"], r["gender"], r["uuid"],
                r["path"], r["dest_path"],
            ])


def split_kodf() -> None:
    """
    Split KoDF into train/test with approximate balance of
    (original/fake × gender) and target total samples.
    """
    cfg = CONFIG_KODF
    random.seed(cfg["seed"])

    original_root = cfg["original_root"]
    others_roots = cfg["others_roots"]
    meta_orig = cfg["metadata_csv_original"]
    meta_fake = cfg["metadata_csv_fake"]
    output_root = cfg["output_root"]

    reset_output_root(output_root)

    print("📂 [KODF] 원본:", original_root)
    print("📂 [KODF] 변조 폴더:", others_roots)
    print("📄 [KODF] 메타(원본):", meta_orig)
    print("📄 [KODF] 메타(변조):", meta_fake)
    print("💾 [KODF] 출력 폴더:", output_root)

    debug_scan_root(original_root, "original_root")
    for i, r in enumerate(others_roots, 1):
        debug_scan_root(r, f"others_root[{i}]")

    orig_uuid, orig_vid = build_gender_map_original(meta_orig)
    fake_uuid, fake_vid = build_gender_map_fake(meta_fake)

    candidates = collect_candidates_kodf(
        original_root, others_roots,
        orig_uuid, orig_vid,
        fake_uuid, fake_vid,
    )

    picked = stratified_sample_kodf(candidates, total=cfg["total"])
    train, test = stratified_split_kodf(picked, train_ratio=cfg["train_ratio"])

    print(f"[KODF] picked={len(picked)}, train={len(train)}, test={len(test)}")

    rows = []

    def handle_split(split_name, arr):
        for s in arr:
            cls = "real" if s["cls"] == "original" else "fake"
            src_path = Path(s["path"])
            dst_dir = output_root / split_name / cls
            ensure_dir(dst_dir)
            dst_path = make_unique(dst_dir / src_path.name)

            shutil.copy2(src_path, dst_path)

            rows.append({
                "split": split_name,
                "cls": cls,
                "gender": s["gender"],
                "uuid": s["uuid"] or "unknown",
                "path": str(src_path),
                "dest_path": str(dst_path),
            })

    handle_split("train", train)
    handle_split("test", test)

    print(f"[KODF] 실제 복사된 영상 수: {len(rows)} 개 (target={cfg['total']})")

    save_manifest_kodf(rows, output_root / "manifest_kodf.csv")
    print("[KODF] split 완료.")


# =========================================================
# PART 3. Frame extraction for deepfake detection
# =========================================================

def collect_all_videos_for_analysis(roots: list[Path]):
    """
    Collect all video paths under multiple root directories.

    This is used to gather split results (FFPP + KoDF) before frame extraction.
    """
    all_videos = []
    for r in roots:
        if not r.exists():
            continue
        for p in iter_videos(r):
            all_videos.append(p)
    print(f"[COLLECT] 총 영상 개수: {len(all_videos)}")
    return all_videos


def extract_frames_for_detection(
    video_paths,
    frames_root: Path,
    frames_per_video: int = 10,
    dataset_roots: dict[str, Path] | None = None,
):
    """
    Extract a fixed number of frames per video for deepfake detection.

    Strategy
    --------
    1) Read FPS of all videos, compute base_fps (most frequent FPS).
    2) For each video, sample frames based on time intervals:
       frame_idx ≈ i * (fps_video / base_fps)
       for i in range(frames_per_video).
    3) Save frames to:
       frames_root / dataset / split / class / video_stem / *.jpg
    4) Save all frame metadata (paths, frame_idx, fps, etc.) to
       frames_manifest.xlsx for later use.

    Parameters
    ----------
    video_paths : list[Path]
        Video files to process.
    frames_root : Path
        Root directory under which to store all extracted frames.
    frames_per_video : int, optional
        Number of frames to extract per video (default = 10).
    dataset_roots : dict[str, Path], optional
        Mapping {dataset_name: dataset_root}.
        Used to infer dataset / split / class from path structure.
    """
    from collections import Counter

    ensure_dir(frames_root)
    if dataset_roots is None:
        dataset_roots = {}

    # 1st pass: gather FPS info and decide base_fps
    fps_list = []
    fps_map = {}   # video_path_str -> fps

    for vp in video_paths:
        vp_str = str(vp)
        cap = cv2.VideoCapture(vp_str)
        if not cap.isOpened():
            print(f"[FRAMES][WARN] open 실패: {vp_str}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            fps_list.append(fps)
            fps_map[vp_str] = fps
        else:
            fps_map[vp_str] = None
        cap.release()

    if not fps_list:
        print("[FRAMES][ERROR] 유효한 FPS가 하나도 없어 프레임 추출을 중단합니다.")
        return

    fps_rounded = [int(round(f)) for f in fps_list if f > 0]
    fps_counter = Counter(fps_rounded)

    print("\n[FRAMES] FPS 종류 및 빈도수:")
    for fps_val in sorted(fps_counter.keys()):
        print(f"  FPS = {fps_val}  →  {fps_counter[fps_val]}개")

    base_fps = fps_counter.most_common(1)[0][0]
    print(f"[FRAMES] 기준(base) FPS = {base_fps} (가장 많이 등장한 FPS)\n")

    # 2nd pass: extract frames for each video
    rows = []  # frame metadata for Excel export

    # Build reverse map: root_path -> dataset_name
    root_to_dataset = {}
    for name, root in dataset_roots.items():
        root_to_dataset[Path(root)] = name

    for vp in video_paths:
        vp_path = Path(vp)
        vp_str = str(vp_path)

        cap = cv2.VideoCapture(vp_str)
        if not cap.isOpened():
            print(f"[FRAMES][WARN] open 실패(2차 패스): {vp_str}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        fps_vid = fps_map.get(vp_str)
        if not fps_vid or fps_vid <= 0:
            cap.release()
            continue

        # Infer dataset / split / class if possible (based on path)
        dataset_name = "UnknownDataset"
        split_name = "unknown_split"
        class_name = "unknown_class"

        for root_path, ds_name in root_to_dataset.items():
            try:
                rel = vp_path.relative_to(root_path)
            except ValueError:
                continue
            else:
                dataset_name = ds_name
                parts = rel.parts  # e.g. ("train", "real", "xxx.mp4")
                if len(parts) >= 3:
                    split_name = parts[0]
                    class_name = parts[1]
                break

        video_stem = vp_path.stem

        # frames_root / dataset / split / class / video_stem
        save_dir = frames_root / dataset_name / split_name / class_name / video_stem
        ensure_dir(save_dir)

        saved_count = 0
        for i in range(frames_per_video):
            frame_idx = int(round(i * fps_vid / base_fps))
            if frame_idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame_name = f"{video_stem}_f{frame_idx:06d}.jpg"
            frame_path = save_dir / frame_name
            cv2.imwrite(str(frame_path), frame)

            rows.append({
                "dataset": dataset_name,
                "split": split_name,
                "class": class_name,
                "video_path": vp_str,
                "video_stem": video_stem,
                "frame_idx": frame_idx,
                "frame_path": str(frame_path),
                "fps_video": fps_vid,
                "base_fps": base_fps,
            })

            saved_count += 1

        cap.release()

    if rows:
        df = pd.DataFrame(rows)
        excel_path = frames_root / "frames_manifest.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"[FRAMES] 총 {len(rows)}개 프레임 메타를 엑셀로 저장했습니다: {excel_path}")
    else:
        print("[FRAMES] 저장된 프레임이 없어 엑셀을 생성하지 않았습니다.")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    # 1) FaceForensics++ split
    split_faceforensics()

    # 2) KoDF split
    split_kodf()

    ffpp_out = CONFIG_FFPP["output_root"]
    kodf_out = CONFIG_KODF["output_root"]

    def count_videos(root: Path) -> int:
        """
        Count how many video files exist under the given root.
        """
        return sum(1 for _ in iter_videos(root))

    print("\n[CHECK] ===== Split 결과 영상 개수 =====")
    print(f"[CHECK] FFPP  : {count_videos(ffpp_out)} 개")
    print(f"[CHECK] KoDF  : {count_videos(kodf_out)} 개")
    print("======================================\n")

    # 3) Collect all split videos and extract 10 frames per video for detection
    all_videos = collect_all_videos_for_analysis([ffpp_out, kodf_out])

    FRAMES_ROOT = Path(r"D:\deoha\Documents\3_kind_Dataset\frames_for_detection")
    extract_frames_for_detection(
        all_videos,
        frames_root=FRAMES_ROOT,
        frames_per_video=10,
        dataset_roots={
            "FFPP": ffpp_out,
            "KoDF": kodf_out,
        },
    )
