import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, lambda_reg=0.03):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

        np.random.seed(42)
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, X_train, y_train, epochs=2000):
        for epoch in range(epochs):
            hidden_layer_input = np.dot(X_train, self.weights_input_hidden)
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            final_input = np.dot(hidden_layer_output, self.weights_hidden_output)
            final_output = self.sigmoid(final_input)

            error = y_train.reshape(-1, 1) - final_output

            d_final_output = error * self.sigmoid_derivative(final_output)
            error_hidden_layer = d_final_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)

            self.weights_hidden_output += hidden_layer_output.T.dot(d_final_output) * self.learning_rate - self.lambda_reg * self.weights_hidden_output
            self.weights_input_hidden += X_train.T.dot(d_hidden_layer) * self.learning_rate - self.lambda_reg * self.weights_input_hidden

    def predict(self, X_new,thresol):
        X_new = (X_new - np.mean(X_new, axis=0)) / np.std(X_new, axis=0)
        hidden_layer_input = np.dot(X_new, self.weights_input_hidden)
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        final_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        final_output = self.sigmoid(final_input)

        y_pred = (final_output > thresol).astype(int)
        return y_pred

file_path = './dataxuly.csv'  
data = pd.read_csv(file_path)
data.drop([0, 1, 2], inplace=True)  

X = data.drop(columns=['STT', 'target']).values

y = data['target'].values

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f'Số mẫu của tập huấn luyện: {len(X_train)}')
print(f'Số mẫu của tập xác thực: {len(X_val)}')
print(f'Số mẫu của tập kiểm tra: {len(X_test)}')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)  
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

input_size = X_train.shape[1]  
hidden_size = 10  
output_size = 1  

nn = NeuralNetwork(input_size, hidden_size, output_size)

nn.train(X_train_resampled, y_train_resampled)

y_train_pred = nn.predict(X_train_resampled,0.5)

train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
print(f'Dộ chính xác trên tập huấn luyện: {train_accuracy:.10f}')

y_val_pred = nn.predict(X_val,0.5)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Dộ chính xác trên tập xác thực: {val_accuracy:.10f}')

y_test_pred = nn.predict(X_test,0.57)  
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Dộ chính xác trên tập kiểm tra: {test_accuracy:.10f}')

y_test_proba = nn.sigmoid(np.dot(nn.sigmoid(np.dot(X_test, nn.weights_input_hidden)), nn.weights_hidden_output)).flatten()  

if len(set(y)) == 2:
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    auc = roc_auc_score(y_test, y_test_proba)
    print(f'Giá trị AUC: {auc:.2f}')
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(y_test_proba, bins=20, kde=True, color='blue')
plt.title("Distribution of Prediction Probabilities")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.show()

conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Ma trận nhầm lẫn:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Không có bệnh', 'Có bệnh'], yticklabels=['Không có bệnh', 'Có bệnh'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.show()

class_report = classification_report(y_test, y_test_pred)
print('Báo cáo phân loại:')
print(class_report)

new_data = np.array([60,1,0,145,282,0,0,142,1,2.8,1,2,3])

predictions = nn.predict(new_data,0.57)

print("Dự đoán:", predictions)

joblib.dump(nn, 'neural_network_model.pkl')  
