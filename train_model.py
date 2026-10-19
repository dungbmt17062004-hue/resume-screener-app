"""Huấn luyện bộ phân loại domain CV: TF-IDF + LinearSVC (calibrated). Chạy: python train_model.py"""
import json, random, time
from pathlib import Path
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from core.taxonomy import TAXONOMY

ART = Path(__file__).parent / "artifacts"
ROLES = {
    "Software Engineering": ["Senior Software Engineer", "Full-Stack Developer", "Backend Engineer", "Frontend Lead"],
    "Data & AI": ["Machine Learning Engineer", "Data Scientist", "MLOps Engineer", "NLP Researcher"],
    "DevOps & Cloud": ["DevOps Engineer", "Cloud Architect", "Site Reliability Engineer", "Platform Engineer"],
    "Product Management": ["Product Manager", "Senior Product Owner", "Technical Product Manager", "Growth PM"],
    "Cybersecurity": ["Security Analyst", "Penetration Tester", "SOC Engineer", "Security Architect"],
}
DUTIES = {
    "Software Engineering": ["Designed and shipped scalable web services", "Refactored legacy modules and improved test coverage", "Built responsive user interfaces and REST endpoints", "Led code reviews and mentored junior developers"],
    "Data & AI": ["Trained and deployed predictive models to production", "Built feature pipelines over large datasets", "Improved model accuracy through experimentation and tuning", "Published experiments and evaluated models offline and online"],
    "DevOps & Cloud": ["Automated infrastructure provisioning and deployments", "Reduced downtime by improving observability and alerting", "Managed container clusters and release pipelines", "Hardened cloud environments and optimized cost"],
    "Product Management": ["Defined product vision and prioritized the roadmap", "Ran discovery with customers and aligned stakeholders", "Launched features and measured impact with experiments", "Wrote requirements and coordinated engineering and design"],
    "Cybersecurity": ["Investigated alerts and led incident response", "Performed security assessments and reported vulnerabilities", "Designed security controls and monitored threats", "Drove audits and security awareness programs"],
}

def synth(domain: str, rng: random.Random) -> str:
    own = rng.sample(TAXONOMY[domain], rng.randint(7, 13))
    other = [rng.choice(TAXONOMY[d]) for d in TAXONOMY if d != domain for _ in range(rng.randint(0, 1))]
    skills = own + rng.sample(other, min(len(other), rng.randint(0, 2)))
    rng.shuffle(skills)
    duties = rng.sample(DUTIES[domain], 3)
    years = rng.randint(1, 12)
    return (f"{rng.choice(ROLES[domain])}\nSummary: {years} years of experience in {domain.lower()}.\n"
            f"Experience\n- " + "\n- ".join(f"{d} using {rng.choice(own)} and {rng.choice(own)}." for d in duties) +
            f"\nSkills\n{', '.join(skills)}\nEducation\nBSc in Computer Science")

def build_dataset(n_per_class=300, seed=42):
    rng = random.Random(seed)
    X, y = [], []
    for d in TAXONOMY:
        for _ in range(n_per_class):
            X.append(synth(d, rng)); y.append(d)
    return X, y

def train(n_per_class=300) -> dict:
    ART.mkdir(exist_ok=True)
    X, y = build_dataset(n_per_class)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english", min_df=2)),
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
    ])
    t0 = time.time(); pipe.fit(Xtr, ytr); fit_s = time.time() - t0
    pred = pipe.predict(Xte)
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    joblib.dump(pipe, ART / "resume_classifier.pkl")
    (ART / "domain_skills.json").write_text(json.dumps(TAXONOMY, indent=2))
    meta = {"accuracy": round(rep["accuracy"], 4), "train_size": len(Xtr), "test_size": len(Xte),
            "vocab_size": len(pipe.named_steps["tfidf"].vocabulary_), "classes": list(pipe.classes_),
            "fit_seconds": round(fit_s, 2), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "report": {k: {m: round(v, 3) for m, v in rep[k].items()} for k in pipe.classes_},
            "note": "Dữ liệu synthetic: độ chính xác cao không đảm bảo hiệu năng trên CV thật."}
    (ART / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta

if __name__ == "__main__":
    m = train()
    print(f"accuracy={m['accuracy']} vocab={m['vocab_size']} fit={m['fit_seconds']}s")
