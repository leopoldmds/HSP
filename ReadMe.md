# TP : Implémentation d'un CNN  - LeNet-5 sur GPU

## 1. Objectifs & méthodes de travail 

### 1.1 Implémentation d'un CNN

#### LeNet-5

##### L'objectif à terme de ces 4 séances est d'implémenter l'inférence dun CNN très classique : LeNet-5

<a href="https://zupimages.net/viewer.php?id=22/02/cqff.png"><img src="https://zupimages.net/up/22/02/cqff.png" alt="" /></a>

## 2. Partie 1 - Prise en main de Cuda : Multiplication de matrices

### Multiplication de matrices

#### Création d'une matrice sur CPU
##### Initialisation d'une matrice de taille n x p, avec des valeurs aléatoires entre -1 et 1
```
void MatrixInit(float *M, int n, int p)
```

##### Initialisation d'une matrice de taille n x p, avec des valeurs égales à 0
```
void MatrixInit0(float *M, int n, int p)
```

#### Affichage d'une matrice sur CPU
```
void MatrixPrint(float *M, int n, int p)
```

#### Addition de deux matrices M1 et M2 de même taille n x p sur CPU
```
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p)
```

#### Addition de deux matrices M1 et M2 de taille n x p sur GPU
```
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p)
```

#### Multiplication de deux matrices M1 et M2 de taille NxN sur CPU
```
void MatrixMult(float *M1, float *M2, float *Mout, int n)
```

#### Multiplication de deux matrices M1 et M2 de taille NxN sur GPU
```
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n)
}
```

## 3. Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

### L'architecture du réseau LeNet-5 est composé de plusieurs couches :
* #### Layer 1 - Génération des données de test
```
MatrixInit0(raw_data,N,P,1); # avec N= P =32
```
* #### Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.
```
__global__ void cudaConv2D(double *img, double *kernels, double * out, int n, int p, int q, int k )
```
* #### Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.
#### Layer 3 - Sous-échantillonnage
```
__global__ void cudaMeanPool(double *in, double *out, int n, int p, int q)
```
#### Tests

### Fonctions d'activation
#### Tanh
```
__device__ double cudaActivationTanh(double val)
```
#### Softmax
```
void ActivationSoftmax(double* input, size_t size)
```
## 4. Partie 3 - Un peu de Python
