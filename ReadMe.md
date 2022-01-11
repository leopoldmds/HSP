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
'void MatrixInit(float *M, int n, int p){
    for (int i=0; i<=n*p-1; i++){
        // M[i]=(float)rand()/RAND_MAX*2-1; // entre -1 et 1
        M[i]=(float)rand()/RAND_MAX; // entre 0 et 1
    }
}'

##### Initialisation d'une matrice de taille n x p, avec des valeurs égales à 0
'void MatrixInit0(float *M, int n, int p){
    for (int i=0; i<=n*p-1; i++){
        M[i]=0;
    }
}'

#### Affichage d'une matrice sur CPU
`void MatrixPrint(float *M, int n, int p){
    printf("  %f  ", M[0]);
    for (int i=1; i<=n*p-1; i++){
        if(i%p==0){
            printf("\n");
        }
        printf("  %f  ", M[i]);
    }
    printf("\n");
}`

#### Addition de deux matrices sur CPU
##### Addition de deux matrices M1 et M2 de même taille n x p
`void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for(int i=0;i<n*p-1;i++){
        Mout[i]=M1[i]+M2[i];
    }
}
`

#### Addition de deux matrices sur GPU


#### Multiplication de deux matrices NxN sur CPU


#### Multiplication de deux matrices NxN sur GPU
