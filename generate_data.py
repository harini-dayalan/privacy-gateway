import json
import random
import os
from datetime import date, timedelta

FIRST_NAMES = ["James", "Maria", "David", "Sarah", "Michael", "Emily", "Robert", "Linda", "William", "Barbara",
               "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Karen", "Charles", "Lisa", "Christopher", "Nancy"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
              "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson"]
DEPARTMENTS = ["Engineering", "HR", "Finance", "Marketing", "Legal", "Operations", "Sales", "IT", "Research", "Clinical"]
CONDITIONS = ["Hypertension", "Type 2 Diabetes", "Asthma", "Arthritis", "Migraine", "Anxiety Disorder",
              "Hypothyroidism", "GERD", "Chronic Back Pain", "Depression"]

def random_dob():
    start = date(1955, 1, 1)
    end = date(2000, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()

def random_ssn():
    return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"

def random_phone():
    return f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"

def random_email(first, last):
    return f"{first.lower()}.{last.lower()}{random.randint(1,99)}@corp-internal.com"

records = []
for i in range(100):
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    record = {
        "id": f"rec_{i+1:04d}",
        "type": random.choice(["HR", "Clinical"]),
        "content": (
            f"Employee/Patient: {first} {last}. "
            f"SSN: {random_ssn()}. "
            f"Date of Birth: {random_dob()}. "
            f"Phone: {random_phone()}. "
            f"Email: {random_email(first, last)}. "
            f"Department: {random.choice(DEPARTMENTS)}. "
            f"Salary: ${random.randint(45000,180000):,}. "
            f"Condition: {random.choice(CONDITIONS)}. "
            f"Notes: Routine review completed for {first} {last} on {date.today().isoformat()}."
        )
    }
    records.append(record)

os.makedirs("data", exist_ok=True)
with open("data/sample_records.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"[generate_data] Created {len(records)} records -> data/sample_records.json")