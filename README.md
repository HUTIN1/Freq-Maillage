# Transformation spectrale appliqué au maillage

Environement Python: 
- pytorch
- tqdm
- pywavefront
- numpy
- pltotly

## Organisation du répartoire github

### Laplacian
Le fichier laplacian.py contient tout les fonctions liées à la transformation spectrale.
[Adjency_matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) construit la matrice adjence du maillage en donnant le nombre de sommet et le faces du maillage. \
[Degree_matrix](https://en.wikipedia.org/wiki/Degree_matrix) construit la matrice des degrés à l'aide du nombre de sommet et les faces du maillage. \
[Graph_laplacian](https://en.wikipedia.org/wiki/Laplacian_matrix) construit le graph laplacian matrice normaizé avec en entrer les sommets et les faces du maillage. \
[Smooth_laplacian](https://en.wikipedia.org/wiki/Laplacian_smoothing) construit le laplacian smooth matice avec en entrer les sommets et les faces du maillage.\
[laplacian_cotangent](https://fr.wikipedia.org/wiki/Op%C3%A9rateur_de_Laplace-Beltrami) construit le laplacian Beltrami matrice avec en enter les sommets et les faces du maillage. \
[Diffusion_maps](https://en.wikipedia.org/wiki/Diffusion_map) construit la diffusion maps du maillage en foncton des sommets et de faces du maillage. \
[MeshReconstruction](https://people.csail.mit.edu/wangyu9/publications/extrinsic-operators/paper.pdf) (fig.4 du paper) reconstruit un maillage en fonction du nombre de valeur propre et fonction propre qu'on veut souhaite. Prend en entrée les sommets, les faces et le nombre de valeur propre utilisé pour la reconstruction.


### Display
Le fichier display possède une fonction d'affichage utilisant la bibliothèque plotly, permettant d'afficher un maillage 3D.

La fonction de display prend en entrer les sommets, les faces et en option les couleurs des triangles du maillage. L'affichage s'effectue directement dans votre navigateur.

### Utils
Le fichier utils possède des fonctions secondaire qui peuvent être réutiliser n'importe où dans le code.

La class OFFReader lit les fichiers .off pour en extraire les sommets et les faces du modéle 3D.

La fonction Readobj prend en entrée le chemin d'un modéle 3D avec l'extension .off ou .obj. Et retourne les sommets et les faces du maillage.

La fonction inv_pow calcul l'inverse de la puissance des coefficient de la matrice. Et prend en compte le problème de inf et de 0. La fonction prend en entrer une matrice et un flotant.


### Main
Le fichier main à pour vocation à être utliser pour implémenter ce que l'on souhaite avec les fonctions crée dans les différents fichier.