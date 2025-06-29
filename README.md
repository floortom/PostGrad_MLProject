## Hyperparameter Search

This project implements two hyperparameter search algorithms for machine learning models: **Grid Search** and **Random Search**.

---

### 🔍 Algorithms

#### 1. Grid Search
- **Exhaustive exploration** of all hyperparameter combinations defined in a predefined grid.  
- **Deterministic**: guarantees every combination is evaluated.  
- **Pros**:
  - Complete coverage of the grid  
  - Reproducible results  
- **Cons**:
  - Computationally expensive for large search spaces  
  - Runtime grows exponentially with the number of hyperparameters  

#### 2. Random Search
- **Random sampling** of hyperparameter combinations from a defined search space.  
- **Flexible**: may discover good configurations by chance.  
- **Pros**:
  - More efficient when hyperparameter count is high  
  - Controls total number of evaluations directly  
- **Cons**:
  - Does not guarantee full coverage of the search space  
  - Results can vary between runs  

---

### ⚖️ When to Use Which

| Criterion                  | Grid Search                         | Random Search                          |
|----------------------------|-------------------------------------|----------------------------------------|
| Coverage                   | Complete                            | Partial (random)                       |
| Determinism                | Yes                                 | No                                     |
| Computational Cost         | High (large grids)                  | Lower (controlled budget)              |
| Best for                   | Small to medium-sized grids         | High-dimensional parameter spaces      |

---

### 📝 Summary

- **Grid Search** is ideal when you need exhaustive, reproducible evaluations of a small-to-moderate hyperparameter grid.  
- **Random Search** is recommended for large or high-dimensional spaces where computational efficiency is crucial.  
- **Choosing between them** depends on your problem size, the dimensionality of your hyperparameter space, and available compute resources.

> _Tip:_ Start with Random Search to quickly identify promising regions, then refine with Grid Search if needed.
