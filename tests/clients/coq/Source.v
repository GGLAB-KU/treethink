From Coq Require Import List.
Import ListNotations.

Theorem rev_app_distr_user :
  forall (A : Type) (xs ys : list A),
    rev (xs ++ ys) = rev ys ++ rev xs.
Admitted.

Theorem rev_involutive_user :
  forall (A : Type) (xs : list A),
    rev (rev xs) = xs.
Admitted.
