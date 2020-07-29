{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Data.Tensor.Shape where

import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import GHC.Base
  ( quotInt,
    remInt,
  )
import GHC.TypeLits
  ( KnownNat,
    Nat,
    natVal,
  )

infixr 5 ><

data m >< n

data Z = Z
  deriving (Show, Read, Eq, Ord)

infixr 3 :.

data m :. n = !m :. !n
  deriving (Show, Read, Eq, Ord)

class Shape sh where
  --   type Sh sh :: Type
  type Index sh :: Type

  rank :: Proxy sh -> Int
  size :: Proxy sh -> Int
  --   toList :: Proxy sh -> [Int]


  toIndex :: Proxy sh -> Index sh -> Int
  fromIndex :: Proxy sh -> Int -> Index sh

  inShape :: Proxy sh -> Index sh -> Bool
  shapeToString :: Proxy sh -> String

instance Shape Z where
  --   type Sh Z = [Int]
  type Index Z = Z

  {-# INLINE [1] rank #-}
  rank _ = 0

  {-# INLINE [1] size #-}
  size _ = 1

  --   {-# INLINE [1] toList #-}
  --   toList _ = []

  {-# INLINE [1] toIndex #-}
  toIndex _ Z = 0

  {-# INLINE [1] fromIndex #-}
  fromIndex _ _ = Z

  {-# INLINE [1] inShape #-}
  inShape _ Z = True

  {-# INLINE [1] shapeToString #-}
  shapeToString _ = "Z"

instance (Shape sh, KnownNat d) => Shape ((d :: Nat) >< sh) where
  --   type Sh (d >< sh) = Sh sh
  type Index (d >< sh) = Int :. Index sh

  {-# INLINE [1] rank #-}
  rank _ = (rank (Proxy @sh)) + 1

  {-# INLINE [1] size #-}
  size _ = dd * (size (Proxy @sh))
    where
      dd = fromIntegral (natVal (Proxy @d))

  --   {-# INLINE [1] toList #-}
  --   toList _ = dd : (toList (Proxy @sh))
  --     where
  --       dd = fromIntegral (natVal (Proxy @d))

  {-# INLINE [1] toIndex #-}
  toIndex _ (x :. xs) = x + dd * (toIndex (Proxy @sh) xs)
    where
      dd = fromIntegral (natVal (Proxy @d))

  {-# INLINE [1] fromIndex #-}
  fromIndex _ x = r :. (fromIndex (Proxy @sh) (x `quotInt` dd))
    where
      dd = fromIntegral (natVal (Proxy @d))
      r
        | rank (Proxy @sh) == 0 = x
        | otherwise = x `remInt` dd

  {-# INLINE [1] inShape #-}
  inShape _ (x :. xs)
    | x >= 0 && x < dd = (inShape (Proxy @sh) xs)
    | otherwise = False
    where
      dd = fromIntegral (natVal (Proxy @d))

  {-# INLINE [1] shapeToString #-}
  shapeToString _ = dd ++ "×" ++ (shapeToString (Proxy @sh))
    where
      dd = show (natVal (Proxy @d))
