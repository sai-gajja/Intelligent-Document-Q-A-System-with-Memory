# quick_real_test.py
import requests

def quick_test():
    # Upload a document and ask a question
    test_content = """
    COMPANY POLICIES DOCUMENT
    
    Vacation Policy:
    Employees are entitled to 15 vacation days per year. Vacation requests must be submitted at least 2 weeks in advance.
    
    Remote Work Policy:
    Employees may work remotely up to 3 days per week with manager approval. 
    Remote workers must be available during core business hours 9 AM - 3 PM.
    
    Expense Policy:
    All business expenses over $100 require pre-approval. Expense reports must be submitted within 30 days.
    
    Performance Reviews:
    Annual performance reviews are conducted in December. Employees should prepare self-assessments one week prior.
    """
    
    # Save and upload
    with open("policies.txt", "w") as f:
        f.write(test_content)
    
    with open("policies.txt", "rb") as f:
        files = {"file": ("policies.txt", f, "text/plain")}
        response = requests.post("http://localhost:8000/upload-document", files=files)
        print(f"Upload: {response.status_code}")
    
    # Ask questions
    questions = [
        "How many vacation days do employees get?",
        "What is the remote work policy?",
        "When are performance reviews conducted?"
    ]
    
    for q in questions:
        response = requests.post("http://localhost:8000/query", json={
            "query": q,
            "session_id": "test123"
        })
        result = response.json()
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print()

if __name__ == "__main__":
    quick_test()