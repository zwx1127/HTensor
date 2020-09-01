{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}

module SVM where

import Data.List
import Data.Proxy (Proxy (..))
import Data.Tensor.Eval
import Data.Tensor.Linear.Base
import Data.Tensor.Linear.Delay
import qualified Data.Tensor.Linear.Seq as S
import qualified Data.Tensor.Operators.Par as P
import qualified Data.Tensor.Operators.Seq as S
import Data.Tensor.Shape
import Data.Tensor.Source
import Data.Tensor.Source.Delay
import Data.Tensor.Source.Unbox
import qualified Data.Vector.Unboxed as U
import GHC.TypeLits (KnownNat, natVal)
import System.Random
import Control.Monad.Trans
import Control.Monad.Trans.State

repeatB :: (Shape sh, U.Unbox a, Floating a) => a -> Tensor D sh a
repeatB = repeatTensor

genK ::
  ( Source r1 a,
    Source r2 a,
    KnownNat m,
    KnownNat n,
    U.Unbox a,
    Floating a
  ) =>
  Vector r1 n a ->
  ( forall r3 r4.
    ( Source r3 a,
      Source r4 a
    ) =>
    ( Vector r3 n a -> Vector r4 n a -> a
    )
  ) ->
  Matrix r2 m n a ->
  Vector D m a
genK x k trainX = generate (\(i' :. Z) -> k (row trainX i') x)

rbf ::
  ( Source r1 a,
    Source r2 a,
    KnownNat n,
    U.Unbox a,
    Floating a
  ) =>
  a ->
  Vector r1 n a ->
  Vector r2 n a ->
  a
rbf a' xi xj = exp $ -a' * (S.normSq (xi |-| xj))

svm' ::
  forall r1 r2 r3 r4 a m n mo.
  ( Source r1 a,
    Source r2 a,
    Source r3 a,
    Source r4 a,
    KnownNat m,
    KnownNat n,
    U.Unbox a,
    Floating a,
    Monad mo
  ) =>
  Vector r1 n a ->
  Matrix r2 m n a ->
  Vector r3 m a ->
  ( forall r5 r6.
    ( Source r5 a,
      Source r6 a
    ) =>
    ( Vector r5 n a -> Vector r6 n a -> a
    )
  ) ->
  Vector r4 m a ->
  a ->
  mo a
svm' x trainX trainY k a' b' = do 
  rst <- P.sumAll $ a' |⊙| trainY |⊙| (genK x k trainX) 
  return $ rst + b'

svm ::
  forall r1 r2 r3 r4 a p m n mo.
  ( Source r1 a,
    Source r2 a,
    Source r3 a,
    Source r4 a,
    KnownNat p,
    KnownNat m,
    KnownNat n,
    U.Unbox a,
    Floating a,
    Monad mo
  ) =>
  Matrix r1 p n a ->
  Matrix r2 m n a ->
  Vector r3 m a ->
  ( forall r5 r6.
    ( Source r5 a,
      Source r6 a
    ) =>
    ( Vector r5 n a -> Vector r6 n a -> a
    )
  ) ->
  Vector r4 m a ->
  a ->
  mo (Vector U p a)
svm x trainX trainY k a' b' = do
  computeP $ dResult |+| (repeatB b')
  where
    dResult :: Vector D p a
    dResult = generate (\(i' :. Z) -> S.sumAll $ a' |⊙| trainY |⊙| (genK (row x i') k trainX))

data SMOST r m a where
  SMOST ::
    (Source r a, KnownNat m) =>
    { a :: Vector r m a,
      b :: a
    } ->
    SMOST r m a

smo ::
  forall r1 r2 a m n mo.
  ( Source r1 a,
    Source r2 a,
    KnownNat m,
    KnownNat n,
    Ord a,
    U.Unbox a,
    Floating a,
    MonadIO mo
  ) =>
  Int ->
  Matrix r1 m n a ->
  Vector r2 m a ->
  ( forall r3 r4.
    ( Source r3 a,
      Source r4 a
    ) =>
    ( Vector r3 n a -> Vector r4 n a -> a
    )
  ) ->
  a ->
  mo (SMOST U m a)
smo n trainX trainY k c = do
  let a' = repeatTensor 0 :: Vector U m a
  let st = SMOST {a=a', b=0}
  kMat <- computeK
  smost <- evalStateT (step n trainX trainY k kMat c) st
  return smost
  where
    computeK :: mo (Matrix U m m a)
    computeK = do
      let kMatD = generate (\(ixi :. ixj :. Z) -> k (row trainX ixi) (row trainX ixj)) :: Matrix D m m a
      computeP kMatD

step :: forall r1 r2 r5 m n a mo. 
  ( Source r1 a, 
    Source r2 a, 
    Source r5 a, 
    KnownNat m, 
    KnownNat n, 
    Ord a, 
    U.Unbox a, 
    Floating a,
    MonadIO mo
  ) => 
  Int -> 
  Matrix r1 m n a -> 
  Vector r2 m a -> 
  ( forall r3 r4. 
    ( Source r3 a, Source r4 a) => 
      ( Vector r3 n a -> Vector r4 n a -> a)
  ) 
  -> Matrix r5 m m a -> 
  a -> 
  StateT (SMOST U m a) mo (SMOST U m a)
step n trainX trainY k kMat c 
  | n == 0 = do
    st <- get
    return st 
  | otherwise = do 
    SMOST{a=a', b=b'} <- get 
    g <- liftIO $ newStdGen
    let (iN, jN) = randCouple 0 (mLength - 1) g
    pi' <- svm' (row trainX iN) trainX trainY k a' b'
    pj' <- svm' (row trainX jN) trainX trainY k a' b'
    let ei = pi' - (trainY !? (iN :. Z))
    let ej = pj' - (trainY !? (jN :. Z))
    aN <- updateA iN jN trainY ei ej kMat a' c
    bN <- updateB iN jN trainY ei ej kMat aN a' b'
    put SMOST{a=aN, b=bN}
    step (n - 1) trainX trainY k kMat c
    where
      mLength :: Int
      mLength = fromInteger $ natVal $ Proxy @m

      randCouple :: (RandomGen g) => Int -> Int -> g -> (Int, Int)
      randCouple l h g = 
        let randArr = take 2 . nub $ (randomRs (l, h) g) :: [Int] in (randArr !! 0, randArr !! 1)

updateA :: 
 forall r1 r2 r3 m a mo.
  ( Source r1 a,
    Source r2 a,
    Source r3 a,
    KnownNat m,
    Ord a,
    U.Unbox a,
    Floating a,
    Monad mo
  ) =>
  Int ->
  Int ->
  Vector r1 m a ->
  a -> a ->
  Matrix r2 m m a ->
  Vector r3 m a ->
  a ->
  mo (Vector U m a)
updateA i' j' trainY ei ej k a' c = do
  let ai = a' !? (i' :. Z)
  let aj = a' !? (j' :. Z)
  let yi = trainY !? (i' :. Z)
  let yj = trainY !? (j' :. Z)
  let kii = k !? (i' :. i' :. Z)
  let kjj = k !? (j' :. j' :. Z)
  let kij = k !? (i' :. j' :. Z)
  let eta = kii + kjj - 2 * kij
  if eta  <= 0
    then do 
      computeP $ delay a'
  else do
    let ajNU = aj + (yj * (ei - ej)) / eta
    let l | yi /= yj = max 0 (aj - ai)
          | otherwise = max 0 (ai + aj - c)
    let h | yi /= yj = min c (c + aj - ai)
          | otherwise = min c (ai + aj)
    let ajN = clip l h ajNU
    let aiN = ai + yi * yj * (aj - ajN)
    computeP (( generate ( \case (ixi :. Z) | ixi == i' -> aiN | ixi == j' -> ajN | otherwise -> a' !? (ixi :. Z))) :: Vector D m a)
  where
    clip :: (Ord a) => a -> a -> a -> a
    clip l h aU
      | aU > h = h
      | aU < l = l
      | otherwise = aU

updateB ::
  ( Source r1 a,
    Source r2 a,
    Source r3 a,
    Source r4 a,
    KnownNat m,
    U.Unbox a,
    Floating a,
    Monad mo
  ) =>
  Int ->
  Int ->
  Vector r1 m a ->
  a -> a ->
  Matrix r2 m m a ->
  Vector r3 m a ->
  Vector r4 m a ->
  a ->
  mo a
updateB i' j' trainY ei ej k aN aO b' = do
  let yi = trainY !? (i' :. Z)
  let yj = trainY !? (j' :. Z)
  let kii = k !? (i' :. i' :. Z)
  let kij = k !? (i' :. j' :. Z)
  let kji = k !? (j' :. i' :. Z)
  let kjj = k !? (j' :. j' :. Z)
  let aiN = aN !? (i' :. Z)
  let ajN = aN !? (j' :. Z)
  let aiO = aO !? (i' :. Z)
  let ajO = aO !? (j' :. Z)
  let biN = -ei - yi * kii * (aiN - aiO) - yj * kji * (ajN - ajO) + b'
  let bjN = -ej - yi * kij * (aiN - aiO) - yj * kjj * (ajN - ajO) + b'
  return $ (biN + bjN) / 2

trainX' :: Matrix U 17 2 Float
trainX' = fromList $
  [ 0.697,
    0.460,
    0.774,
    0.376,
    0.634,
    0.264,
    0.608,
    0.318,
    0.556,
    0.215,
    0.403,
    0.237,
    0.481,
    0.149,
    0.437,
    0.211,
    0.666,
    0.091,
    0.243,
    0.267,
    0.245,
    0.057,
    0.343,
    0.099,
    0.639,
    0.161,
    0.657,
    0.198,
    0.360,
    0.370,
    0.593,
    0.042,
    0.719,
    0.103
  ]

trainY' :: Vector U 17 Float
trainY' = fromList $ (take 8 (repeat 1.0)) <> (take 9 (repeat (-1.0)))

testSVM :: IO ()
testSVM = do
  let alpha = 5
  let c = 20
  SMOST{a=aN, b=bN} <- smo 100000 trainX' trainY' (rbf alpha) c
  print aN
  print bN
  result <- svm trainX' trainX' trainY' (rbf alpha) aN bN
  print result