"""
🎖️ SUPPORT VECTORS REVEALED: Who Controls the Cancer Boundary?
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

print("🎖️ SUPPORT VECTORS: THE VIP PATIENTS DEMO")
print("=" * 50)

# Load cancer data
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"📊 Total training patients: {len(X_train)}")
print(f"📊 Total test patients: {len(X_test)}")
print()

# Train LINEAR SVM (easier to understand)
print("🧪 Training LINEAR SVM...")
linear_svm = svm.SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

# Get predictions
y_pred = linear_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.3f}")
print()

# 🎯 THE MAGIC REVEAL: Support Vectors!
print("🎖️ SUPPORT VECTORS ANALYSIS:")
print("-" * 30)

total_support_vectors = len(linear_svm.support_)
total_training = len(X_train)
percentage = (total_support_vectors / total_training) * 100

print(f"🎯 Total training patients: {total_training}")
print(f"🎖️ Support vectors (VIPs): {total_support_vectors}")
print(f"📊 Percentage who are VIPs: {percentage:.1f}%")
print()

# Show support vectors per class
support_vectors_per_class = linear_svm.n_support_
print(f"🔴 Malignant support vectors: {support_vectors_per_class[0]}")
print(f"🟢 Benign support vectors: {support_vectors_per_class[1]}")
print()

print("🤯 MIND-BLOWING FACTS:")
print(f"   • Only {total_support_vectors} out of {total_training} patients control the ENTIRE boundary!")
print(f"   • That's just {percentage:.1f}% of patients!")
print(f"   • If you removed the other {total_training - total_support_vectors} patients, the boundary wouldn't change!")
print()

# 🧪 Compare different kernels and their support vectors
print("🎭 KERNEL COMPARISON - WHO NEEDS MORE VIPs?")
print("-" * 40)

kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    if kernel == 'poly':
        clf = svm.SVC(kernel=kernel, degree=3, random_state=42)
    else:
        clf = svm.SVC(kernel=kernel, random_state=42)
    
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    support_count = len(clf.support_)
    support_pct = (support_count / total_training) * 100
    
    print(f"🎯 {kernel.upper():6}: {support_count:3d} VIPs ({support_pct:4.1f}%) | Accuracy: {acc:.3f}")

print()
print("🔍 INSIGHT:")
print("   • Different kernels need different numbers of support vectors!")
print("   • More complex kernels (RBF, POLY) often need more VIPs")
print("   • But sometimes simpler is better (like your LINEAR results)!")
print()

print("🎉 CONGRATULATIONS!")
print("You've just discovered which patients are the TRUE decision-makers!")
print("These support vectors are the most important cases for medical AI! 🏥✨")