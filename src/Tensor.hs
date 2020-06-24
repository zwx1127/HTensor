{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Tensor where

import           Data.Kind                      ( Type )
import           Data.Proxy                     ( Proxy(..) )
import           GHC.TypeLits                   ( KnownNat
                                                , Nat
                                                , natVal
                                                )

infixr 5 ><
data (m :: Nat) >< (n :: Nat)

class HasDim a where
  type Dim a :: Type
  toArray :: [Int] -> Proxy a -> [Int]

instance (KnownNat d) => HasDim (d :: Nat) where
  type Dim d = [Int]
  toArray xs _ = (fromIntegral (natVal (Proxy @d))) : xs

instance (HasDim a, KnownNat d) => HasDim ((d :: Nat) >< a) where
  type Dim (d >< a) = Dim a
  toArray xs _ = toArray ((fromIntegral (natVal (Proxy @d))) : xs) (Proxy @a)

dimToArray :: (HasDim a) => Proxy a -> [Int]
dimToArray = toArray []

data Tensor d a where
  Tensor ::(HasDim d, Num a) => [a] -> Tensor d a

showTensor :: forall d a . (Show a, HasDim d, Num a) => Tensor d a -> String
showTensor (Tensor xs) =
  "(" <> showDim (dimToArray (Proxy @d)) <> ")" <> "\n" <> (show xs)
 where
  showDim :: [Int] -> String
  showDim (x : []) = show x
  showDim (x : xs) = show x <> "×" <> showDim xs
  showDim _        = error "incorrect dimension."

instance (Show a, HasDim d, Num a) => Show (Tensor d a) where
  show = showTensor

infixr 6 !!!

(!!!) :: forall d a . (HasDim d, Num a) => Tensor d a -> [Int] -> a
(!!!) (Tensor xs) i = xs !! (index (dimToArray (Proxy @d)) i)
 where
  index :: [Int] -> [Int] -> Int
  index (s : []) (i : []) | i < s     = i
                          | otherwise = error "index not match shap."
  index (s : ss) (i : is) | i < s     = s * i + (index ss is)
                          | otherwise = error "index not match shap."
  index _ _ = error "index not match shap."

fromArray :: forall d a . (HasDim d, Num a) => [a] -> Tensor d a
fromArray xs = Tensor @d xs

mapTensor :: (HasDim d, Num a) => (a -> a) -> Tensor d a -> Tensor d a
mapTensor f (Tensor xs) = fromArray (fmap f xs)
