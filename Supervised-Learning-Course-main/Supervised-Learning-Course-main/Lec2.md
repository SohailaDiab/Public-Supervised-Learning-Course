## Structured Data vs Unstructured Data
### - Structured Data:
  - Refers to data that is organized in a pre-defined format with a specific schema or model.
  - Has a clearly defined data model that makes it easy to search, analyze, and manipulate the data.
  - Examples of structured data include spreadsheets, databases, and XML files.
  - Is usually easy to process with the help of database management systems (DBMS), data warehouses, and analytics tools.
  - Can be analyzed using various analytical techniques like data mining, machine learning, and statistical analysis.
### - Unstructured Data:
  - Refers to data that is not organized in a predefined format or schema.
  - Does not follow a specific data model or schema and is usually not easily searchable.
  - Examples of unstructured data include emails, social media posts, images, videos, and audio recordings.
  - Is often more difficult to process than structured data, as it requires advanced text analytics, natural language processing (NLP), and machine learning techniques.
  - Can be analyzed using techniques such as sentiment analysis, image and video analysis, and topic modeling.
  
  
## Features and Dimensions
- 1 feature -> 1-D space
- 2 features -> 2-D space
- 3 features -> 3-D space
- n features -> n-D space

## Dot Product
![image](https://user-images.githubusercontent.com/70928356/229141893-1fddc95b-1640-4bcb-b541-833ea3d3cdcc.png)

- Can be used as a measure of similarity
The dot product of 2 similar vectors (normalized) is very high
  
## Determinant
![image](https://user-images.githubusercontent.com/70928356/229144487-7aacc6da-de2d-4409-bcc8-49cf04cdbea5.png)

- The change of the area after the matrix has changed.
- It is a scalar value that can be computed from a square matrix.
- It provides information about the matrix's invertibility, its eigenvalues, and the scaling factor of linear transformations represented by the matrix.
- The determinant is denoted by the symbol "det" or "|".
- It is defined for a square matrix A of size n x n as a sum of products of matrix elements that satisfies certain rules.
- The purpose of the determinant is to provide a measure of how much the matrix A "stretches" or "shrinks" space.
- If the determinant is zero, the matrix is singular and the linear transformation represented by the matrix does not preserve volume or orientation.
- If the determinant is nonzero, the matrix is invertible and the linear transformation preserves volume and orientation.
- The absolute value of the determinant provides a measure of the scaling factor of the transformation.

### Singular Matrix
- Determinant = 0
- Its inverse is undefined.
- It is a square matrix (m = n) that is not invertible.

. . .

- If determinant of matrix A is 0, it means that the area of A is going to a null space (0). So, we cannot go back. Therefore, the singular matrix has no inverse and its det=0.
- It is like the concept of singularity of black holes.


## Dependancy
- If we a column that is dependent on others... eg. we have columns A, B and C, and 2A+B=C ... then it is a **singular** matrx. 
- In other words, if we can get a column by doin a linear comination of other columns.

## Rank
- The rank of a matrix is a measure of its "dimensionality".
- Specifically, the rank of a matrix is the number of linearly independent rows or columns it contains.
- A set of rows or columns is linearly independent if none of the rows or columns can be expressed as a linear combination of the others.
### Full rank matrix
- All rows and all columns are independent.
- It is important in ML. If all columns (features) are independent, this means all of the features are important to take into account when predicting.
- We can get the inverse of this matrix, which is important to have.

## Matrix Inverse
- It is important to perform the division operation on a matrix... A/B = AB^-1.
- If we have a transformation that we multiply by the inverse of itself, we will get an **Identity matrix** (a neutral matrix).

## Sine and Cosine (lec 2 part 2 -- this specific part is around min 17:00 in video)
![image](https://user-images.githubusercontent.com/70928356/229248967-e8b170d5-827f-4798-a173-84eeff31f545.png)

## Similarity (End of lec 2 part 2)
![image](https://user-images.githubusercontent.com/70928356/229250378-494a7310-edd3-4db3-9d15-bdeda55f63c3.png)

