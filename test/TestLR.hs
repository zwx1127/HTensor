{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module TestLR (testLR) where

import Data.Tensor.Eval
import Data.Tensor.Linear.Base
import Data.Tensor.Linear.Par ((|⋅|))
import Data.Tensor.Linear.Delay
import Data.Tensor.Operators.Delay
import Data.Tensor.Source.Delay
import Data.Tensor.Source
import Data.Tensor.Source.Unbox
import qualified Data.Vector.Unboxed as U
import GHC.TypeLits
  ( KnownNat,
  )

-- Logistic Regression
predict ::
  ( Source r1 e,
    Source r2 e,
    KnownNat m,
    KnownNat n,
    U.Unbox e,
    Floating e
  ) =>
  Vector r1 m e ->
  Matrix r2 n m e ->
  IO (Matrix U n 1 e)
predict wb x' = do
  computeP $ mapTensor sigmoid (x' |*| (cv wb))

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (- x))

loss ::
  forall r1 r2 m n e mo.
  ( Source r1 e,
    Source r2 e,
    KnownNat m,
    KnownNat n,
    U.Unbox e,
    Floating e,
    Monad mo
  ) =>
  Vector r1 m e ->
  Matrix r2 n m e ->
  Vector U n e ->
  mo e
loss wb x' y =
  let z = x' |*| (cv wb)
      repeat1 = (repeatTensor 1) :: RVector D n e
   in repeat1 |⋅| (((cv y) |⊙| z) |+| (mapTensor (\z' -> log ((exp z')) + 1) z))

-- Gradient Descent
gradient ::
  forall r1 r2 r3 e m n.
  ( Source r1 e,
    Source r2 e,
    Source r3 e,
    KnownNat m,
    KnownNat n,
    Floating e
  ) =>
  Vector r1 m e ->
  Matrix r2 n m e ->
  Vector r3 n e ->
  Vector D m e
gradient wb x' y = reshape $ (transpose x') |*| ((cv y) |-| (mapTensor p (x' |*| (cv wb))))
  where
    p :: (Floating a) => a -> a
    p x = 1 / (1 + (exp (- x)))

gd ::
  forall r1 r2 r3 e m n mo.
  ( Source r1 e,
    Source r2 e,
    Source r3 e,
    KnownNat m,
    KnownNat n,
    U.Unbox e,
    Floating e,
    Monad mo
  ) =>
  e ->
  Vector r1 m e ->
  Matrix r2 n m e ->
  Vector r3 n e ->
  mo (Vector U m e)
gd rate wb x' y = do
  let d = wb |+| (rate .*| (gradient wb x' y))
  computeP d

train ::
  forall r m n e mo.
  ( Source r e,
    KnownNat m,
    KnownNat n,
    U.Unbox e,
    Floating e,
    Monad mo
  ) =>
  Int ->
  e ->
  Vector r m e ->
  Matrix r n m e ->
  Vector r n e ->
  mo (Vector U m e)
train i rate wb x' y
  | i == 0 = gd rate wb x' y
  | otherwise = do
    wb' <- train (i - 1) rate wb x' y
    gd rate wb' x' y

-- -- Newton's method
-- newton ::
--   forall r1 r2 r3 m n e mo.
--   ( Source r1 e,
--     Source r2 e,
--     Source r3 e,
--     KnownNat m,
--     KnownNat n,
--     U.Unbox e,
--     Floating e,
--     Monad mo
--   ) =>
--   Vector r1 m e ->
--   Matrix r2 n m e ->
--   Vector r3 n e ->
--   mo (Vector U m e)
-- newton wb x' y = do
--   let p = (mapTensor (\z -> (exp z) / (1 + exp z)) (x' |*| (cv wb)))
--       p' = mapTensor (\x -> x * (1 - x)) p
--       x'' = transpose x'
--   secD <- ((rv (diagonal (x' |*| x''))) |⋅| p')
--   computeP $ wb |+| (((1 / secD) .*| (x'' |*| ((cv y) |-| p))) `column` 0)

-- trainNewton ::
--   ( Source r1 e,
--     Source r2 e,
--     Source r3 e,
--     KnownNat m,
--     KnownNat n,
--     U.Unbox e,
--     Floating e,
--     Monad mo
--   ) =>
--   Int ->
--   Vector r1 m e ->
--   Matrix r2 n m e ->
--   Vector r3 n e ->
--   mo (Vector U m e)
-- trainNewton i wb x' y
--   | i == 0 = newton wb x' y
--   | otherwise = do
--     wb' <- trainNewton (i - 1) wb x' y
--     newton wb' x' y

-- data set
trainX :: [Double]
trainX =
  [ 0.697,
    0.460,
    1,
    0.774,
    0.376,
    1,
    0.634,
    0.264,
    1,
    0.608,
    0.318,
    1,
    0.556,
    0.215,
    1,
    0.403,
    0.237,
    1,
    0.481,
    0.149,
    1,
    0.437,
    0.211,
    1,
    0.666,
    0.091,
    1,
    0.243,
    0.267,
    1,
    0.245,
    0.057,
    1,
    0.343,
    0.099,
    1,
    0.639,
    0.161,
    1,
    0.657,
    0.198,
    1,
    0.360,
    0.370,
    1,
    0.593,
    0.042,
    1,
    0.719,
    0.103,
    1
  ]

trainY :: [Double]
trainY = (take 8 (repeat 1.0)) <> (take 9 (repeat 0.0))

trainWB :: [Double]
trainWB = [0, 0, 0]

testLR :: IO ()
testLR = do
  let trainX' = (fromList trainX) :: Matrix U 15 3 Double
  r <- train 80000 0.01 (fromList trainWB) trainX' (fromList trainY)
  lossr <- loss r trainX' (fromList trainY)
  print lossr
  pr <- predict r trainX'
  print r
  print pr
