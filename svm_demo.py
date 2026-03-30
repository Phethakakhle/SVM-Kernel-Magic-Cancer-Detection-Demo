# Step 1: Install required packages (run this in terminal if needed)
# pip install scikit-learn pandas matplotlib

# Step 2: Import everything we need
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

print("🏥 CANCER DETECTION: KERNEL MAGIC DEMO")
print("=" * 50)

# Step 3: Load the real breast cancer dataset
print("📊 Loading breast cancer dataset...")
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target  # 0=malignant, 1=benign

print(f"📈 Dataset Info:")
print(f"   - Total patients: {len(X)}")
print(f"   - Features per patient: {X.shape[1]}")
print(f"   - Malignant cases: {sum(y==0)}")
print(f"   - Benign cases: {sum(y==1)}")
print()

# Step 4: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("🎭 THE GREAT KERNEL FACE-OFF!")
print("-" * 30)

# Step 5: Test different kernels
kernels = ['linear', 'rbf', 'poly']
results = {}

for kernel_name in kernels:
    print(f"🧪 Testing {kernel_name.upper()} kernel...")
    
    # Create and train the SVM
    if kernel_name == 'poly':
        clf = svm.SVC(kernel=kernel_name, degree=3, random_state=42)
    else:
        clf = svm.SVC(kernel=kernel_name, random_state=42)
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[kernel_name] = accuracy
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"   ✅ Accuracy: {accuracy:.3f}")
    print(f"   📊 Confusion Matrix:")
    print(f"       [[{cm[0,0]:2d} {cm[0,1]:2d}]  ← Predictions")
    print(f"        [{cm[1,0]:2d} {cm[1,1]:2d}]]  ← Actual")
    print()

# Step 6: Show the magic results!
print("🎯 FINAL RESULTS - KERNEL MAGIC REVEALED!")
print("=" * 50)

best_kernel = max(results, key=results.get)
for kernel, accuracy in results.items():
    emoji = "🏆" if kernel == best_kernel else "📊"
    magic = " ← WINNER!" if kernel == best_kernel else ""
    print(f"{emoji} {kernel.upper():10} kernel: {accuracy:.3f}{magic}")

print()
print("🤯 WHAT JUST HAPPENED?")
print(f"   • LINEAR:     Drew straight lines through {X.shape[1]}D space")
print(f"   • RBF:        Created magical similarity 'bubbles' ✨") 
print(f"   • POLYNOMIAL: Found complex feature interactions 🔄")
print()
print("🎉 You just witnessed the kernel trick in action!")

# Step 7: Optional - Create a simple visualization
try:
    # Create a comparison chart
    plt.figure(figsize=(10, 6))
    
    # Bar chart of accuracies
    plt.subplot(1, 2, 1)
    colors = ['skyblue', 'lightgreen', 'salmon']
    bars = plt.bar(results.keys(), results.values(), color=colors)
    plt.title('SVM Kernel Performance Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0.85, 1.0)
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars, results.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.subplot(1, 2, 2)
    plt.pie(results.values(), labels=[k.upper() for k in results.keys()], 
            autopct='%1.1f%%', colors=colors)
    plt.title('Relative Performance')
    
    plt.tight_layout()
    plt.show()
    
    print("📈 Visualization created! Check your output for the charts.")
    
except Exception as e:
    print("📊 Visualization skipped (matplotlib not available)")

print("\n🎓 CONGRATULATIONS!")
print("You've successfully run SVMs with different kernels!")
print("Try changing the kernel parameters and see what happens! 🚀")