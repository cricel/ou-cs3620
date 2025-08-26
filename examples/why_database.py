import csv
import os
import random
import sqlite3
import string
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


# -------------------------
# Data model (synthetic)
# -------------------------
@dataclass
class Student:
    student_id: int
    full_name: str


@dataclass
class Course:
    course_id: int
    title: str


def generate_students(num_students: int, seed: int = 42) -> List[Student]:
    random.seed(seed)
    students: List[Student] = []
    for sid in range(1, num_students + 1):
        name = "".join(random.choices(string.ascii_uppercase, k=6))
        students.append(Student(student_id=sid, full_name=f"Student {name}"))
    return students


def generate_courses(num_courses: int, seed: int = 1337) -> List[Course]:
    random.seed(seed)
    courses: List[Course] = []
    for cid in range(1, num_courses + 1):
        title = "".join(random.choices(string.ascii_uppercase, k=5))
        courses.append(Course(course_id=cid, title=f"Course {title}"))
    return courses


def generate_enrollments(
    students: List[Student],
    courses: List[Course],
    avg_enrollments_per_student: int = 5,
    seed: int = 7,
) -> List[Tuple[int, int]]:
    random.seed(seed)
    enrollments: List[Tuple[int, int]] = []
    course_ids = [c.course_id for c in courses]
    for s in students:
        # Poisson-like variability using randint around avg
        count = max(0, int(random.gauss(mu=avg_enrollments_per_student, sigma=2)))
        chosen = random.sample(course_ids, k=min(count, len(course_ids)))
        for cid in chosen:
            enrollments.append((s.student_id, cid))
    return enrollments


# -------------------------
# In-memory Python approach
# -------------------------
class PythonInMemoryStore:
    def __init__(self, students: List[Student], courses: List[Course], enrollments: List[Tuple[int, int]]):
        self.student_id_to_student: Dict[int, Student] = {s.student_id: s for s in students}
        self.course_id_to_course: Dict[int, Course] = {c.course_id: c for c in courses}

        # Build an index to simulate a relational link: student_id -> [course_id]
        self.student_id_to_course_ids: Dict[int, List[int]] = defaultdict(list)
        for sid, cid in enrollments:
            self.student_id_to_course_ids[sid].append(cid)

        # For aggregation demo: course popularity counts
        self.course_id_to_enrollment_count: Dict[int, int] = defaultdict(int)
        for _sid, cid in enrollments:
            self.course_id_to_enrollment_count[cid] += 1

    def get_courses_for_student(self, student_id: int) -> List[str]:
        titles: List[str] = []
        for cid in self.student_id_to_course_ids.get(student_id, []):
            course = self.course_id_to_course.get(cid)
            if course:
                titles.append(course.title)
        return titles

    def get_top_n_popular_courses(self, n: int) -> List[Tuple[str, int]]:
        # Sort by count descending, then map to titles
        top = sorted(self.course_id_to_enrollment_count.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [(self.course_id_to_course[cid].title, count) for cid, count in top]


# -------------------------
# SQLite approach (relational)
# -------------------------
class SQLiteStore:
    def __init__(self, students: List[Student], courses: List[Course], enrollments: List[Tuple[int, int]], db_path: str = ":memory:"):
        # If a file path is provided, rebuild the database file fresh
        if db_path != ":memory:" and os.path.exists(db_path):
            os.remove(db_path)

        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=off;")
        self.conn.execute("PRAGMA synchronous=off;")
        self.conn.execute("PRAGMA temp_store=memory;")
        self._init_schema()
        self._bulk_insert(students, courses, enrollments)
        self._create_indexes()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE students (
                student_id INTEGER PRIMARY KEY,
                full_name  TEXT NOT NULL
            );
            CREATE TABLE courses (
                course_id INTEGER PRIMARY KEY,
                title     TEXT NOT NULL
            );
            CREATE TABLE enrollments (
                student_id INTEGER NOT NULL,
                course_id  INTEGER NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(student_id),
                FOREIGN KEY(course_id)  REFERENCES courses(course_id)
            );
            """
        )
        cur.close()

    def _bulk_insert(self, students: List[Student], courses: List[Course], enrollments: List[Tuple[int, int]]) -> None:
        cur = self.conn.cursor()
        cur.executemany("INSERT INTO students(student_id, full_name) VALUES (?, ?)", [(s.student_id, s.full_name) for s in students])
        cur.executemany("INSERT INTO courses(course_id, title) VALUES (?, ?)", [(c.course_id, c.title) for c in courses])
        cur.executemany("INSERT INTO enrollments(student_id, course_id) VALUES (?, ?)", enrollments)
        self.conn.commit()
        cur.close()

    def _create_indexes(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE INDEX idx_enrollments_student ON enrollments(student_id);
            CREATE INDEX idx_enrollments_course  ON enrollments(course_id);
            """
        )
        self.conn.commit()
        cur.close()

    def get_courses_for_student(self, student_id: int) -> List[str]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT c.title
            FROM enrollments e
            JOIN courses c ON c.course_id = e.course_id
            WHERE e.student_id = ?
            """,
            (student_id,),
        )
        rows = cur.fetchall()
        cur.close()
        return [r[0] for r in rows]

    def get_courses_for_many_students(self, student_ids: Iterable[int]) -> List[Tuple[int, str]]:
        # Batch query to leverage indexes and avoid per-student overhead
        ids = list(student_ids)
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        sql = f"""
            SELECT e.student_id, c.title
            FROM enrollments e
            JOIN courses c ON c.course_id = e.course_id
            WHERE e.student_id IN ({placeholders})
        """
        cur = self.conn.cursor()
        cur.execute(sql, ids)
        rows = cur.fetchall()
        cur.close()
        return [(sid, title) for (sid, title) in rows]

    def get_top_n_popular_courses(self, n: int) -> List[Tuple[str, int]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT c.title, COUNT(*) AS enroll_count
            FROM enrollments e
            JOIN courses c ON c.course_id = e.course_id
            GROUP BY e.course_id
            ORDER BY enroll_count DESC
            LIMIT ?
            """,
            (n,),
        )
        rows = cur.fetchall()
        cur.close()
        return [(title, count) for (title, count) in rows]


# -------------------------
# Benchmark helpers
# -------------------------
def time_operation(label: str, fn, *args, iterations: int = 1, **kwargs) -> Tuple[str, float]:
    start = time.perf_counter()
    last_result = None
    for _ in range(iterations):
        last_result = fn(*args, **kwargs)
    duration = time.perf_counter() - start
    print(f"{label}: {duration:.4f}s")
    return label, duration


def write_csv_files(data_dir: str, students: List[Student], courses: List[Course], enrollments: List[Tuple[int, int]]) -> None:
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "students.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "full_name"]) 
        for s in students:
            w.writerow([s.student_id, s.full_name])
    with open(os.path.join(data_dir, "courses.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["course_id", "title"]) 
        for c in courses:
            w.writerow([c.course_id, c.title])
    with open(os.path.join(data_dir, "enrollments.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "course_id"]) 
        for sid, cid in enrollments:
            w.writerow([sid, cid])


class PythonFileStore:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def _iter_csv(self, name: str) -> Iterable[List[str]]:
        with open(os.path.join(self.data_dir, name), newline="") as f:
            r = csv.reader(f)
            header_skipped = False
            for row in r:
                if not header_skipped:
                    header_skipped = True
                    continue
                yield row

    def get_courses_for_many_students(self, student_ids: Iterable[int]) -> List[Tuple[int, str]]:
        target = set(student_ids)
        # Scan enrollments once; collect for target student_ids
        sid_to_cids: Dict[int, List[int]] = defaultdict(list)
        for sid_str, cid_str in self._iter_csv("enrollments.csv"):
            sid = int(sid_str)
            if sid in target:
                sid_to_cids[sid].append(int(cid_str))

        # Build set of all needed course ids
        needed_cids = set(cid for cids in sid_to_cids.values() for cid in cids)

        # Load courses for needed ids only
        cid_to_title: Dict[int, str] = {}
        for cid_str, title in self._iter_csv("courses.csv"):
            cid = int(cid_str)
            if cid in needed_cids:
                cid_to_title[cid] = title

        results: List[Tuple[int, str]] = []
        for sid, cids in sid_to_cids.items():
            for cid in cids:
                title = cid_to_title.get(cid)
                if title is not None:
                    results.append((sid, title))
        return results

    def get_top_n_popular_courses(self, n: int) -> List[Tuple[str, int]]:
        # Count enrollments per course by single scan
        counts: Dict[int, int] = defaultdict(int)
        for _sid_str, cid_str in self._iter_csv("enrollments.csv"):
            counts[int(cid_str)] += 1

        # Load titles for courses that appear at least once
        cid_to_title: Dict[int, str] = {}
        needed = set(counts.keys())
        for cid_str, title in self._iter_csv("courses.csv"):
            cid = int(cid_str)
            if cid in needed:
                cid_to_title[cid] = title

        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [(cid_to_title[cid], cnt) for cid, cnt in top if cid in cid_to_title]


def run_benchmark(num_students: int = 500_000, num_courses: int = 20_000, avg_enrollments_per_student: int = 5, use_file_backed: bool = True) -> None:
    print("Generating synthetic data...")
    students = generate_students(num_students)
    courses = generate_courses(num_courses)
    enrollments = generate_enrollments(students, courses, avg_enrollments_per_student)

    print(f"Students: {len(students):,} | Courses: {len(courses):,} | Enrollments: {len(enrollments):,}")

    if use_file_backed:
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        data_dir = os.path.join(tmp_dir, "data")
        print(f"\nWriting CSV files to {data_dir}...")
        t0 = time.perf_counter()
        write_csv_files(data_dir, students, courses, enrollments)
        print(f"Wrote files in {time.perf_counter() - t0:.4f}s")

        print("Building SQLite FILE database with indexes...")
        t0 = time.perf_counter()
        sql_store = SQLiteStore(students, courses, enrollments, db_path=os.path.join(tmp_dir, "demo.db"))
        print(f"Built DB in {time.perf_counter() - t0:.4f}s")

        py_file_store = PythonFileStore(data_dir)
    else:
        print("\nBuilding Python in-memory indexes...")
        t0 = time.perf_counter()
        py_store = PythonInMemoryStore(students, courses, enrollments)
        print(f"Built in {time.perf_counter() - t0:.4f}s")

        print("Building SQLite in-memory database with indexes...")
        t0 = time.perf_counter()
        sql_store = SQLiteStore(students, courses, enrollments)
        print(f"Built in {time.perf_counter() - t0:.4f}s")

    # Choose random sample of student_ids for repeated lookups
    sample_size = 5_000
    random.seed(99)
    sample_student_ids = random.sample([s.student_id for s in students], k=min(sample_size, len(students)))

    print("\nQuery: Get courses for many students (joins)")
    if use_file_backed:
        def py_file_many(ids: Iterable[int]) -> int:
            return len(py_file_store.get_courses_for_many_students(ids))

        def sql_many(ids: Iterable[int]) -> int:
            return len(sql_store.get_courses_for_many_students(ids))

        _, py_join_time = time_operation("Python FILE (CSV scan)", py_file_many, sample_student_ids)
        _, sql_join_time = time_operation("SQLite FILE (indexed batch JOIN)", sql_many, sample_student_ids)
    else:
        def py_many_queries(ids: Iterable[int]) -> int:
            total = 0
            for sid in ids:
                total += len(py_store.get_courses_for_student(sid))
            return total

        def sql_many_queries(ids: Iterable[int]) -> int:
            # Use batched query for fairness
            return len(sql_store.get_courses_for_many_students(ids))

        _, py_join_time = time_operation("Python (pre-indexed dicts)", py_many_queries, sample_student_ids)
        _, sql_join_time = time_operation("SQLite (indexed batch JOIN)", sql_many_queries, sample_student_ids)

    print("\nQuery: Top 10 popular courses (aggregation)")
    if use_file_backed:
        def py_file_topn() -> List[Tuple[str, int]]:
            return py_file_store.get_top_n_popular_courses(10)

        def sql_topn() -> List[Tuple[str, int]]:
            return sql_store.get_top_n_popular_courses(10)

        _, py_top_time = time_operation("Python FILE (scan + aggregate)", py_file_topn)
        _, sql_top_time = time_operation("SQLite FILE (GROUP BY + ORDER BY)", sql_topn)
    else:
        def py_topn() -> List[Tuple[str, int]]:
            return py_store.get_top_n_popular_courses(10)

        def sql_topn() -> List[Tuple[str, int]]:
            return sql_store.get_top_n_popular_courses(10)

        _, py_top_time = time_operation("Python (manual aggregation)", py_topn)
        _, sql_top_time = time_operation("SQLite (GROUP BY + ORDER BY)", sql_topn)

    print("\nSample result (Top 3 courses):")
    if use_file_backed:
        sample_py = py_file_topn()
        sample_sql = sql_topn()
    else:
        sample_py = py_topn()
        sample_sql = sql_topn()
    print("Python:", sample_py[:3])
    print("SQLite:", sample_sql[:3])

    print("\nSummary (lower is faster):")
    print(f"- Get courses for many students -> Python: {py_join_time:.4f}s | SQLite: {sql_join_time:.4f}s")
    print(f"- Top 10 popular courses       -> Python: {py_top_time:.4f}s | SQLite: {sql_top_time:.4f}s")


def main() -> None:
    run_benchmark()


if __name__ == "__main__":
    main()
