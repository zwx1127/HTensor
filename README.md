# HTensor
Haskell type-level multi-dimensional array

Base on [repa](https://github.com/haskell-repa/repa), Implement type-level multi-dimensional array and basic linear algebra algorithm.

For more information see the tests.

# Usage
## Tensor
``` haskell
-- create from list
t0 :: Tensor U (3 >< 4 >< Z) Int
t0 = fromList [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

-- create from function
t1 :: Tensor U (3 >< 4 >< Z) Int
t1 = generate (\(x :. y :. Z) -> x + y)
```
## Linear
Every operator has Seq (sequence), Par (parallel) Delay (delayed tensor structure) three versions, except the reduce operator such as the fold, dot product.
``` haskell
-- create matrix
m :: Matrix U 3 4 Int
m = fromList [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

-- Seq transpose
m' :: Matrix U 4 3 Int
m' = transpose m

-- Par matmul
main :: IO ()
main = do
  r <- m |*| m' :: IO (Matrix U 3 3 Int)
  print t

-- Delay matmul
r :: Matrix D 3 3 Int
r = m |*| m'
```