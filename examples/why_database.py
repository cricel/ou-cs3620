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
# These dataclasses represent our core entities in a simple way
# In a real application, these would be more complex with additional fields
@dataclass
class Student:
    student_id: int
    full_name: str


@dataclass
class Course:
    course_id: int
    title: str


def generate_students(num_students: int, seed: int = 42) -> List[Student]:
    """Generate synthetic student data for benchmarking.
    
    Args:
        num_students: Number of students to generate
        seed: Random seed for reproducible results
    
    Returns:
        List of Student objects with sequential IDs and random names
    """
    random.seed(seed)
    students: List[Student] = []
    for sid in range(1, num_students + 1):
        # Generate a random 6-character name for each student
        name = "".join(random.choices(string.ascii_uppercase, k=6))
        students.append(Student(student_id=sid, full_name=f"Student {name}"))
    return students


def generate_courses(num_courses: int, seed: int = 1337) -> List[Course]:
    """Generate synthetic course data for benchmarking.
    
    Args:
        num_courses: Number of courses to generate
        seed: Random seed for reproducible results
    
    Returns:
        List of Course objects with sequential IDs and random titles
    """
    random.seed(seed)
    courses: List[Course] = []
    for cid in range(1, num_courses + 1):
        # Generate a random 5-character title for each course
        title = "".join(random.choices(string.ascii_uppercase, k=5))
        courses.append(Course(course_id=cid, title=f"Course {title}"))
    return courses


def generate_enrollments(
    students: List[Student],
    courses: List[Course],
    avg_enrollments_per_student: int = 5,
    seed: int = 7,
) -> List[Tuple[int, int]]:
    """Generate synthetic enrollment data (many-to-many relationship).
    
    This creates realistic enrollment patterns where:
    - Each student enrolls in a variable number of courses
    - The number follows a normal distribution around the average
    - Students can't enroll in more courses than exist
    
    Args:
        students: List of students to enroll
        courses: List of available courses
        avg_enrollments_per_student: Average number of courses per student
        seed: Random seed for reproducible results
    
    Returns:
        List of (student_id, course_id) tuples representing enrollments
    """
    random.seed(seed)
    enrollments: List[Tuple[int, int]] = []
    course_ids = [c.course_id for c in courses]
    
    for s in students:
        # Generate variable enrollment count using normal distribution
        # This creates realistic variability (some students take more/less courses)
        count = max(0, int(random.gauss(mu=avg_enrollments_per_student, sigma=2)))
        # Randomly select courses for this student (without duplicates)
        chosen = random.sample(course_ids, k=min(count, len(course_ids)))
        for cid in chosen:
            enrollments.append((s.student_id, cid))
    return enrollments


# -------------------------
# In-memory Python approach
# -------------------------
# This class demonstrates how you might handle relational data using pure Python
# It pre-builds indexes (dictionaries) to speed up lookups, similar to database indexes
class PythonInMemoryStore:
    def __init__(self, students: List[Student], courses: List[Course], enrollments: List[Tuple[int, int]]):
        # Create lookup dictionaries for fast access by ID
        # This is like having primary key indexes in a database
        self.student_id_to_student: Dict[int, Student] = {s.student_id: s for s in students}
        self.course_id_to_course: Dict[int, Course] = {c.course_id: c for c in courses}

        # Build a "join index" - maps student_id to list of their course_ids
        # This simulates what a database does with foreign key relationships
        self.student_id_to_course_ids: Dict[int, List[int]] = defaultdict(list)
        for sid, cid in enrollments:
            self.student_id_to_course_ids[sid].append(cid)

        # Pre-compute aggregation: count enrollments per course
        # This is like a materialized view or pre-computed summary
        self.course_id_to_enrollment_count: Dict[int, int] = defaultdict(int)
        for _sid, cid in enrollments:
            self.course_id_to_enrollment_count[cid] += 1

    def get_courses_for_student(self, student_id: int) -> List[str]:
        """Get all course titles for a specific student (simulates a JOIN query)."""
        titles: List[str] = []
        # Use our pre-built index to get course IDs for this student
        for cid in self.student_id_to_course_ids.get(student_id, []):
            # Look up the course title using our course index
            course = self.course_id_to_course.get(cid)
            if course:
                titles.append(course.title)
        return titles

    def get_top_n_popular_courses(self, n: int) -> List[Tuple[str, int]]:
        """Get the N most popular courses by enrollment count (simulates GROUP BY + ORDER BY)."""
        # Sort our pre-computed counts and take top N
        top = sorted(self.course_id_to_enrollment_count.items(), key=lambda kv: kv[1], reverse=True)[:n]
        # Map course IDs back to titles
        return [(self.course_id_to_course[cid].title, count) for cid, count in top]


# -------------------------
# SQLite approach (relational)
# -------------------------
# This class demonstrates using a proper relational database
# It creates tables, relationships, and indexes just like a real database would
class SQLiteStore:
    def __init__(self, students: List[Student], courses: List[Course], enrollments: List[Tuple[int, int]], db_path: str = ":memory:"):
        # For file-based databases, start fresh each time for consistent benchmarking
        if db_path != ":memory:" and os.path.exists(db_path):
            os.remove(db_path)

        # Connect to SQLite (in-memory or file-based)
        self.conn = sqlite3.connect(db_path)
        
        # Optimize SQLite for bulk operations and speed
        # These settings trade durability for speed (ok for benchmarking)
        self.conn.execute("PRAGMA journal_mode=off;")  # Disable WAL journaling
        self.conn.execute("PRAGMA synchronous=off;")   # Disable fsync calls
        self.conn.execute("PRAGMA temp_store=memory;") # Keep temp tables in memory
        
        # Set up the database structure and data
        self._init_schema()
        self._bulk_insert(students, courses, enrollments)
        self._create_indexes()

    def _init_schema(self) -> None:
        """Create the database schema with proper relationships."""
        cur = self.conn.cursor()
        cur.executescript(
            """
            -- Students table (like a real student database)
            CREATE TABLE students (
                student_id INTEGER PRIMARY KEY,
                full_name  TEXT NOT NULL
            );
            
            -- Courses table (like a course catalog)
            CREATE TABLE courses (
                course_id INTEGER PRIMARY KEY,
                title     TEXT NOT NULL
            );
            
            -- Enrollments table (many-to-many relationship)
            -- This is the "join table" that connects students to courses
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
        """Insert all data into the database using bulk operations for speed."""
        cur = self.conn.cursor()
        
        # Bulk insert students (much faster than individual INSERT statements)
        cur.executemany("INSERT INTO students(student_id, full_name) VALUES (?, ?)", 
                       [(s.student_id, s.full_name) for s in students])
        
        # Bulk insert courses
        cur.executemany("INSERT INTO courses(course_id, title) VALUES (?, ?)", 
                       [(c.course_id, c.title) for c in courses])
        
        # Bulk insert enrollments (the relationship data)
        cur.executemany("INSERT INTO enrollments(student_id, course_id) VALUES (?, ?)", enrollments)
        
        self.conn.commit()
        cur.close()

    def _create_indexes(self) -> None:
        """Create database indexes to speed up queries.
        
        Indexes are like the 'lookup tables' we built in PythonInMemoryStore,
        but they're managed by the database engine and optimized for disk storage.
        """
        cur = self.conn.cursor()
        cur.executescript(
            """
            -- Index on student_id for fast lookups of "what courses does this student take?"
            CREATE INDEX idx_enrollments_student ON enrollments(student_id);
            
            -- Index on course_id for fast lookups of "what students take this course?"
            CREATE INDEX idx_enrollments_course  ON enrollments(course_id);
            """
        )
        self.conn.commit()
        cur.close()

    def get_courses_for_student(self, student_id: int) -> List[str]:
        """Get all course titles for a specific student using SQL JOIN.
        
        This is equivalent to the Python version but uses SQL to express the relationship.
        The database engine handles the join optimization using our indexes.
        """
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
        """Get courses for multiple students in a single query (batch operation).
        
        This demonstrates a key advantage of databases: you can query multiple
        records efficiently in one operation, rather than making many individual queries.
        """
        ids = list(student_ids)
        if not ids:
            return []
        
        # Build a SQL query with placeholders for all student IDs
        # This is like: WHERE student_id IN (1, 2, 3, 4, 5, ...)
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
        """Get the N most popular courses using SQL aggregation.
        
        This demonstrates SQL's GROUP BY and ORDER BY capabilities.
        The database engine handles the counting, grouping, and sorting efficiently.
        """
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
    """Time a function execution and print the result.
    
    Args:
        label: Description of what's being timed
        fn: Function to time
        iterations: Number of times to run the function (for averaging)
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        Tuple of (label, duration_in_seconds)
    """
    start = time.perf_counter()
    last_result = None
    for _ in range(iterations):
        last_result = fn(*args, **kwargs)
    duration = time.perf_counter() - start
    print(f"{label}: {duration:.4f}s")
    return label, duration


def write_csv_files(data_dir: str, students: List[Student], courses: List[Course], enrollments: List[Tuple[int, int]]) -> None:
    """Write data to CSV files for the file-backed Python approach.
    
    This simulates having data stored in flat files (like CSV exports from a database).
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Write students to CSV
    with open(os.path.join(data_dir, "students.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "full_name"])  # Header row
        for s in students:
            w.writerow([s.student_id, s.full_name])
    
    # Write courses to CSV
    with open(os.path.join(data_dir, "courses.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["course_id", "title"])  # Header row
        for c in courses:
            w.writerow([c.course_id, c.title])
    
    # Write enrollments to CSV (the relationship data)
    with open(os.path.join(data_dir, "enrollments.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "course_id"])  # Header row
        for sid, cid in enrollments:
            w.writerow([sid, cid])


class PythonFileStore:
    """Simulates reading data from CSV files (like a data analyst might do).
    
    This approach reads from disk each time, which is slower than keeping
    everything in memory, but more realistic for large datasets that don't fit in RAM.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def _iter_csv(self, name: str) -> Iterable[List[str]]:
        """Read a CSV file row by row, skipping the header."""
        with open(os.path.join(self.data_dir, name), newline="") as f:
            r = csv.reader(f)
            header_skipped = False
            for row in r:
                if not header_skipped:
                    header_skipped = True  # Skip the header row
                    continue
                yield row

    def get_courses_for_many_students(self, student_ids: Iterable[int]) -> List[Tuple[int, str]]:
        """Get courses for multiple students by scanning CSV files.
        
        This simulates what you'd do if you had CSV files and needed to join them:
        1. Read enrollments.csv to find which courses each student takes
        2. Read courses.csv to get the course titles
        3. Combine the data manually (like a manual JOIN operation)
        """
        target = set(student_ids)
        
        # Step 1: Scan enrollments.csv to find courses for target students
        # This is like filtering the enrollments table
        sid_to_cids: Dict[int, List[int]] = defaultdict(list)
        for sid_str, cid_str in self._iter_csv("enrollments.csv"):
            sid = int(sid_str)
            if sid in target:
                sid_to_cids[sid].append(int(cid_str))

        # Step 2: Figure out which course titles we need to look up
        needed_cids = set(cid for cids in sid_to_cids.values() for cid in cids)

        # Step 3: Read courses.csv to get titles for the needed courses
        # This is like joining with the courses table
        cid_to_title: Dict[int, str] = {}
        for cid_str, title in self._iter_csv("courses.csv"):
            cid = int(cid_str)
            if cid in needed_cids:
                cid_to_title[cid] = title

        # Step 4: Combine the data to get (student_id, course_title) pairs
        results: List[Tuple[int, str]] = []
        for sid, cids in sid_to_cids.items():
            for cid in cids:
                title = cid_to_title.get(cid)
                if title is not None:
                    results.append((sid, title))
        return results

    def get_top_n_popular_courses(self, n: int) -> List[Tuple[str, int]]:
        """Get the N most popular courses by counting enrollments in CSV files.
        
        This simulates manual aggregation (like GROUP BY in SQL):
        1. Count how many times each course appears in enrollments
        2. Sort by count to find the most popular
        3. Look up the course titles
        """
        # Step 1: Count enrollments per course by scanning enrollments.csv
        # This is like GROUP BY course_id, COUNT(*) in SQL
        counts: Dict[int, int] = defaultdict(int)
        for _sid_str, cid_str in self._iter_csv("enrollments.csv"):
            counts[int(cid_str)] += 1

        # Step 2: Load course titles for courses that have enrollments
        cid_to_title: Dict[int, str] = {}
        needed = set(counts.keys())
        for cid_str, title in self._iter_csv("courses.csv"):
            cid = int(cid_str)
            if cid in needed:
                cid_to_title[cid] = title

        # Step 3: Sort by count and take top N (like ORDER BY count DESC LIMIT N)
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [(cid_to_title[cid], cnt) for cid, cnt in top if cid in cid_to_title]


def run_benchmark(num_students: int = 500_000, num_courses: int = 20_000, avg_enrollments_per_student: int = 5, use_file_backed: bool = True) -> None:
    """Run the main benchmark comparing different data storage approaches.
    
    This function:
    1. Generates synthetic data (students, courses, enrollments)
    2. Sets up different storage systems (Python vs SQLite)
    3. Runs equivalent queries on both systems
    4. Times and compares the results
    
    Args:
        num_students: Number of students to generate
        num_courses: Number of courses to generate  
        avg_enrollments_per_student: Average courses per student
        use_file_backed: If True, use file-based storage; if False, use in-memory
    """
    print("Generating synthetic data...")
    students = generate_students(num_students)
    courses = generate_courses(num_courses)
    enrollments = generate_enrollments(students, courses, avg_enrollments_per_student)

    print(f"Students: {len(students):,} | Courses: {len(courses):,} | Enrollments: {len(enrollments):,}")

    if use_file_backed:
        # File-backed mode: write to disk and read from files
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
        # In-memory mode: keep everything in RAM
        print("\nBuilding Python in-memory indexes...")
        t0 = time.perf_counter()
        py_store = PythonInMemoryStore(students, courses, enrollments)
        print(f"Built in {time.perf_counter() - t0:.4f}s")

        print("Building SQLite in-memory database with indexes...")
        t0 = time.perf_counter()
        sql_store = SQLiteStore(students, courses, enrollments)
        print(f"Built in {time.perf_counter() - t0:.4f}s")

    # Choose a random sample of students for testing queries
    # This simulates real-world scenarios where you query for specific students
    sample_size = 5_000
    random.seed(99)
    sample_student_ids = random.sample([s.student_id for s in students], k=min(sample_size, len(students)))

    print("\nQuery: Get courses for many students (joins)")
    if use_file_backed:
        # File-backed mode: compare CSV scanning vs SQLite with indexes
        def py_file_many(ids: Iterable[int]) -> int:
            return len(py_file_store.get_courses_for_many_students(ids))

        def sql_many(ids: Iterable[int]) -> int:
            return len(sql_store.get_courses_for_many_students(ids))

        _, py_join_time = time_operation("Python FILE (CSV scan)", py_file_many, sample_student_ids)
        _, sql_join_time = time_operation("SQLite FILE (indexed batch JOIN)", sql_many, sample_student_ids)
    else:
        # In-memory mode: compare pre-built indexes vs SQLite
        def py_many_queries(ids: Iterable[int]) -> int:
            total = 0
            for sid in ids:
                total += len(py_store.get_courses_for_student(sid))
            return total

        def sql_many_queries(ids: Iterable[int]) -> int:
            # Use batched query for fairness (single SQL query vs multiple Python lookups)
            return len(sql_store.get_courses_for_many_students(ids))

        _, py_join_time = time_operation("Python (pre-indexed dicts)", py_many_queries, sample_student_ids)
        _, sql_join_time = time_operation("SQLite (indexed batch JOIN)", sql_many_queries, sample_student_ids)

    print("\nQuery: Top 10 popular courses (aggregation)")
    if use_file_backed:
        # File-backed mode: compare manual aggregation vs SQL GROUP BY
        def py_file_topn() -> List[Tuple[str, int]]:
            return py_file_store.get_top_n_popular_courses(10)

        def sql_topn() -> List[Tuple[str, int]]:
            return sql_store.get_top_n_popular_courses(10)

        _, py_top_time = time_operation("Python FILE (scan + aggregate)", py_file_topn)
        _, sql_top_time = time_operation("SQLite FILE (GROUP BY + ORDER BY)", sql_topn)
    else:
        # In-memory mode: compare pre-computed vs SQL aggregation
        def py_topn() -> List[Tuple[str, int]]:
            return py_store.get_top_n_popular_courses(10)

        def sql_topn() -> List[Tuple[str, int]]:
            return sql_store.get_top_n_popular_courses(10)

        _, py_top_time = time_operation("Python (manual aggregation)", py_topn)
        _, sql_top_time = time_operation("SQLite (GROUP BY + ORDER BY)", sql_topn)

    # Show sample results to verify both approaches give the same answer
    print("\nSample result (Top 3 courses):")
    if use_file_backed:
        sample_py = py_file_topn()
        sample_sql = sql_topn()
    else:
        sample_py = py_topn()
        sample_sql = sql_topn()
    print("Python:", sample_py[:3])
    print("SQLite:", sample_sql[:3])

    # Print final comparison summary
    print("\nSummary (lower is faster):")
    print(f"- Get courses for many students -> Python: {py_join_time:.4f}s | SQLite: {sql_join_time:.4f}s")
    print(f"- Top 10 popular courses       -> Python: {py_top_time:.4f}s | SQLite: {sql_top_time:.4f}s")


def main() -> None:
    """Main entry point - run the database vs Python benchmark."""
    run_benchmark()


if __name__ == "__main__":
    main()
