# training/train.py

from model.classifier import Classifier, ClassifierWrapper
from model.embeddings import EmbeddingModel
from utils.text_cleaner import clean_text
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import ast
from collections import Counter
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configuración
DATASET_PATH = "./datasets/sentfinv1.1.csv"
MODEL_SAVE_PATH = "./model.pt"
EMBEDDING_MODEL = "philschmid/bge-base-financial-matryoshka"
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_prepare_dataset(path: str, verbose: bool = False):
    if verbose:
        print(f"Cargando dataset desde {path}...")
    df = pd.read_csv(path).iloc[:, 1:]
    if verbose:
        print("Aplicando ast.literal_eval a la columna 'Decisions'...")
    df["Decisions"] = df["Decisions"].apply(ast.literal_eval)
    if verbose:
        print("Extrayendo sentimiento mayoritario de 'Decisions'...")
    df["Sentiment"] = df["Decisions"].apply(lambda d: Counter(d.values()).most_common(1)[0][0] if d else "unknown")
    df = df[["Title", "Sentiment"]]
    df.columns = ["text", "sentiment"]
    df = df[df["sentiment"] != "unknown"]

    mapping = {"negative": -1, "neutral": 0, "positive": 1}
    df["sentiment"] = df["sentiment"].map(mapping)
    if verbose:
        print("Limpiando textos...")
    df["text"] = df["text"].apply(clean_text)

    if verbose:
        print(f"Dataset preparado con {len(df)} muestras.")
    return df

def encode_labels(labels: np.ndarray, verbose: bool = False) -> torch.Tensor:
    label_map = {-1: 0, 0: 1, 1: 2}
    if verbose:
        print("Codificando etiquetas...")
    return torch.tensor([label_map[l] for l in labels], dtype=torch.long)

def compute_class_weights(y: np.ndarray, verbose: bool = False) -> torch.Tensor:
    if verbose:
        print("Calculando pesos de clase...")
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y)
    if verbose:
        print(f"Pesos de clase: {weights}")
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

def train_model(X_train, Y_train, X_val, Y_val, embed_dim, verbose: bool = False):
    if verbose:
        print("Inicializando modelo...")
    model = Classifier(emb_dim=embed_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    class_weights = compute_class_weights(Y_train.cpu().numpy(), verbose=verbose)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        if verbose:
            print(f"\n--- Epoch {epoch} ---")
        model.train()
        total_loss = 0
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if verbose and (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}: Loss={loss.item():.4f}")
        train_loss = total_loss / len(train_loader)

        # Validación
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_loss += criterion(logits, yb).item()
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
        val_loss /= len(val_loader)
        val_acc = correct / len(Y_val)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            no_improve = 0
            if verbose:
                print("  Mejorando modelo (guardando estado)...")
        else:
            no_improve += 1
            if verbose:
                print(f"  Sin mejora en la validación ({no_improve}/{PATIENCE})")
            if no_improve >= PATIENCE:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    return model

def save_model(model, save_path, emb_dim, verbose: bool = False):
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_args": {
            "emb_dim": emb_dim,
            "hidden_dim": 256,
            "num_classes": 3,
            "dropout": 0.2
        }
    }, save_path)
    print(f"Modelo guardado en {save_path}")

def main(verbose: bool = True):
    df = load_and_prepare_dataset(DATASET_PATH, verbose=verbose)
    if verbose:
        print("Dividiendo dataset en entrenamiento y validación...")
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        df["text"].tolist(), df["sentiment"].values,
        test_size=0.15, stratify=df["sentiment"], random_state=505
    )

    if verbose:
        print("Inicializando modelo de embeddings...")
    embedder = EmbeddingModel(model_name=EMBEDDING_MODEL, device=DEVICE)
    if verbose:
        print("Codificando textos de entrenamiento...")
    X_train = embedder.encode(X_train_texts)
    if verbose:
        print("Codificando textos de validación...")
    X_val = embedder.encode(X_val_texts)
    Y_train = encode_labels(y_train, verbose=verbose).to(DEVICE)
    Y_val = encode_labels(y_val, verbose=verbose).to(DEVICE)

    model = train_model(X_train, Y_train, X_val, Y_val, embed_dim=X_train.shape[1], verbose=verbose)
    save_model(model, MODEL_SAVE_PATH, X_train.shape[1], verbose=verbose)

def evaluate_model():
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Modelo no encontrado en {MODEL_SAVE_PATH}. Por favor, entrena el modelo primero.")
        return

    if not os.path.exists(DATASET_PATH):
        print(f"Dataset no encontrado en {DATASET_PATH}. Por favor, verifica la ruta.")
        return

    df = load_and_prepare_dataset(DATASET_PATH)
    embedder = EmbeddingModel(model_name=EMBEDDING_MODEL, device=DEVICE)
    X_texts = df["text"].tolist()
    X = embedder.encode(X_texts)
    y = encode_labels(df["sentiment"].values).to(DEVICE)

    model = ClassifierWrapper(model_path=MODEL_SAVE_PATH, device=DEVICE)
    model.model.eval()

    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model.predict_logits(xb)
            pred = logits.argmax(dim=1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\nPrecisión del modelo: {accuracy:.4f}")

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("\nMatriz de confusión (filas: verdadero, columnas: predicho):")
    print("      neg  neu  pos")
    for i, row in enumerate(cm):
        label = ["neg", "neu", "pos"][i]
        print(f"{label:>4} {row[0]:>4} {row[1]:>4} {row[2]:>4}")


if __name__ == "__main__":
    # main(verbose=True)
    evaluate_model()
