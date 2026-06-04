From Coq Require Import List.
Import ListNotations.

Theorem rev_app_distr_user :
  forall (A : Type) (xs ys : list A),
    rev (xs ++ ys) = rev ys ++ rev xs.
Proof.
  intros A xs ys.
  induction xs as [|x xs IH]; simpl.
  - now rewrite app_nil_r.
  - rewrite IH, app_assoc. reflexivity.
Qed.

Theorem rev_involutive_user :
  forall (A : Type) (xs : list A),
    rev (rev xs) = xs.
Proof.
  intros A xs.
  induction xs as [|x xs IH]; simpl.
  - reflexivity.
  - rewrite rev_app_distr_user. simpl. rewrite IH. reflexivity.
Qed.
