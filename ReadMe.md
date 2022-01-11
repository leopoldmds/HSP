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
void MatrixInit(float *M, int n, int p){
    for (int i=0; i<=n*p-1; i++){
        // M[i]=(float)rand()/RAND_MAX*2-1; // entre -1 et 1
        M[i]=(float)rand()/RAND_MAX; // entre 0 et 1
    }
}
```

##### Initialisation d'une matrice de taille n x p, avec des valeurs égales à 0
```
void MatrixInit0(float *M, int n, int p){
    for (int i=0; i<=n*p-1; i++){
        M[i]=0;
    }
}
```

#### Affichage d'une matrice sur CPU
```
void MatrixPrint(float *M, int n, int p){
    printf("  %f  ", M[0]);
    for (int i=1; i<=n*p-1; i++){
        if(i%p==0){
            printf("\n");
        }
        printf("  %f  ", M[i]);
    }
    printf("\n");
}
```

#### Addition de deux matrices M1 et M2 de même taille n x p sur CPU
```
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for(int i=0;i<n*p-1;i++){
        Mout[i]=M1[i]+M2[i];
    }
}
```

#### Addition de deux matrices M1 et M2 de taille n x p sur GPU
```
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    printf("Addition depuis le GPU en cours...\n\n");
    
    for (int i=0; i<n*p; i++){
        Mout[i] = M1[i]+M2[i];
    }
}
```

#### Multiplication de deux matrices M1 et M2 de taille NxN sur CPU
```
void MatrixMult(float *M1, float *M2, float *Mout, int n){    
    for (int lig = 0; lig < n; lig++){
        for (int col = 0; col < n; col++){
            float s = 0.0f;
            for (int i = 0; i < n; i++) {
                s += M1[lig * n + i] * M2[i * n + col];
            }
            Mout[lig * n + col] = s;
        }
    }
}

```

#### Multiplication de deux matrices M1 et M2 de taille NxN sur GPU
```
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    printf("Multiplication depuis le GPU en cours...\n\n");
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float s = 0.0f;
    
    if(lig < n && col < n){
        for (int i = 0; i < n; i++){
            s += M1[lig * n + i] * M2[i * n + col];
        }
    }
    Mout[lig * n + col] = s;
}
```

## 2. Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

##### L'architecture du réseau LeNet-5 est composé de plusieurs couches :
* ##### Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST
* ##### Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.
* ##### Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.

