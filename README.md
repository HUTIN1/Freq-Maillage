# Transformation spectrale appliquée au maillage

Environnement Python: 
- pytorch
- tqdm
- pywavefront
- numpy
- pltotly

## Organisation du répertoire github

### Laplacian
Le fichier laplacian.py contient toutes les fonctions liées à la transformation spectrale.\
[Adjency_matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) construit la matrice adjence du maillage en donnant le nombre de sommets et les faces du maillage.\
[Degree_matrix](https://en.wikipedia.org/wiki/Degree_matrix) construit la matrice des degrés à l'aide du nombre de sommets et les faces du maillage.\
[Graph_laplacian](https://en.wikipedia.org/wiki/Laplacian_matrix) construit le graph laplacien matrice normaizée avec en entrer les sommets et les faces du maillage.\
[Smooth_laplacian](https://en.wikipedia.org/wiki/Laplacian_smoothing) construit le laplacien smooth matice avec en entrer les sommets et les faces du maillage.\
[laplacian_cotangent](https://fr.wikipedia.org/wiki/Op%C3%A9rateur_de_Laplace-Beltrami) construit le laplacien Beltrami matrice avec en enter les sommets et les faces du maillage.\
[Diffusion_maps](https://en.wikipedia.org/wiki/Diffusion_map) construit la diffusion maps du maillage en fonction des sommets et des faces du maillage.\
[MeshReconstruction](https://people.csail.mit.edu/wangyu9/publications/extrinsic-operators/paper.pdf) (fig.4 du paper) reconstruit un maillage en fonction du nombre de valeurs propres et fonction propre qu'on veut souhaite. Prends en entrée les sommets, les faces et le nombre de valeurs propres utilisé pour la reconstruction.

### Display
Le fichier display possède une fonction d'affichage utilisant la bibliothèque plotly, permettant d'afficher un maillage 3D.

La fonction de display prend en entrée les sommets, les faces et en option les couleurs des triangles du maillage. L'affichage s'effectue directement dans votre navigateur.

### Utils
Le fichier utils possède des fonctions secondaires qui peuvent être réutilisées n'importe où dans le code.

La class OFFReader lit les fichiers .off pour en extraire les sommets et les faces du modèle 3D.

La fonction Readobj prend en entrée le chemin d'un modèle 3D avec l'extension .off ou .obj. Et retourne les sommets et les faces du maillage.

La fonction inv_pow calcule l'inverse de la puissance des coefficients de la matrice. Et prend en compte le problème de inf et de 0. La fonction prend en entrée une matrice et un flottant.

### Main
Le fichier main à pour vocation à être utiliser pour implémenter ce que l'on souhaite avec les fonctions crée dans les différents fichiers.
