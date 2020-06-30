{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeOperators #-}

module Neural where

import           GHC.TypeLits                   ( KnownNat )
import           Tensor
import           Linear

data Dense m n a = Dense {
    w :: Matrix n m a,
    x :: Vector m a,
    b :: a,
    f :: (a -> a)
}

infixr 5 :~:

data Neural m n a where
    Neural ::(KnownNat m, KnownNat n, Num a) => Dense m n a -> Neural m n a
    (:~:) ::(KnownNat m, KnownNat n, KnownNat p, Num a) => Neural m p a -> Neural p n a -> Neural m n a





