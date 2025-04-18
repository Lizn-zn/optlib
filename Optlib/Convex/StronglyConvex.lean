/-
Copyright (c) 2023 Chenyi Li. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Chenyi Li, Ziyu Wang
-/
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Convex.Strong
import Optlib.Function.Lsmooth

/-!
  the properties of strongly convex function and gradient descent method
  for strongly convex function
-/
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

variable {s : Set E} {f : E → ℝ} {m : ℝ} {xm xm': E} {f' : E → E} {mp : m > 0}

section Strongly_Convex

open Set InnerProductSpace

theorem Strongly_Convex_Bound (m : ℝ) (strongly_convex: StrongConvexOn s m f):
  ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s →
     ∀ ⦃a⦄, 0 < a → ∀ ⦃b⦄, 0 < b → a + b = 1 → f (a • x + b • y)
       ≤ a * f x + b * f y - m / 2 * a * b * ‖x - y‖ ^ 2 := by
  intro x xs y ys a ap b bp abp
  rcases strongly_convex with ⟨_, sc⟩
  have anneg : a ≥ 0 := by linarith
  have bnneg : b ≥ 0 := by linarith
  rcases sc xs ys anneg bnneg abp with ineq
  simp at ineq
  rw [mul_comm (m / 2), mul_assoc a, mul_comm (m / 2),← mul_assoc, mul_assoc (a * b)]
  apply ineq

theorem stronglyConvexOn_def (hs : Convex ℝ s)
    (hfun : ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s →
      ∀ ⦃a⦄, 0 ≤ a → ∀ ⦃b⦄, 0 ≤ b → a + b = 1 → f (a • x + b • y)
        ≤ a * f x + b * f y - m / 2 * a * b * ‖x - y‖ ^ 2) :
    StrongConvexOn s m f := by
  constructor
  · exact hs
  intro x hx y hy a b ha hb hab
  specialize hfun hx hy ha hb hab
  dsimp
  have : m / 2 * a * b * ‖x - y‖ ^ 2 = a * b * (m / 2 * ‖x - y‖ ^ 2) := by ring_nf
  simp at this;
  rw [← this]; exact hfun

theorem Strongly_Convex_Unique_Minima (hsc: StrongConvexOn s m f) {mp : m > 0}
    (min: IsMinOn f s xm) (min' : IsMinOn f s xm') (hxm : xm ∈ s) (hxm' : xm' ∈ s): xm = xm' := by
  by_contra neq
  push_neg at neq
  have eq : f xm = f xm' := by
    apply le_antisymm
    . apply min hxm'
    . apply min' hxm
  let x := (2 : ℝ)⁻¹ • xm + (2 : ℝ)⁻¹ • xm'
  have xeq : x = (2 : ℝ)⁻¹ • xm + (2 : ℝ)⁻¹ • xm' := by rfl
  rcases hsc with ⟨cr, sc⟩
  have : (0 : ℝ) ≤ 1 / 2 := by norm_num
  have p : (0 : ℝ) < 1 / 2 := by norm_num
  have xs : x ∈ s := by
    rcases convex_iff_forall_pos.mp cr hxm hxm' p p (by norm_num) with cxs
    simp at cxs
    rw [← xeq] at cxs
    apply cxs
  specialize sc hxm hxm' this this (by norm_num)
  simp at sc
  rw [← xeq,← eq] at sc
  rw [← two_mul,← mul_assoc, mul_inv_cancel₀ (by norm_num), one_mul] at sc
  have normp : ‖xm - xm'‖ > 0 := by
    apply norm_sub_pos_iff.mpr
    apply neq
  have nng : m / 2 * ‖xm - xm'‖ ^ 2 > 0 := by
    apply mul_pos
    . linarith
    . apply pow_pos; linarith
  apply absurd (min xs)
  simp [← xeq]
  calc
    f x ≤ f xm - 2⁻¹ * 2⁻¹ * (m / 2 * ‖xm - xm'‖ ^ 2) := by apply sc
    _ < f xm := by apply lt_of_sub_pos; simp; apply nng


/--
  The sum of a convex function and a square of norm is strongly convex
-/

lemma strongconvex_of_convex_add_sq (f : E → ℝ) (x : E) (hfun : ConvexOn ℝ univ f) :
    StrongConvexOn univ (1 : ℝ) fun u ↦ f u + ‖u - x‖ ^ 2 / 2 := by
  rw [ConvexOn] at hfun
  rcases hfun with ⟨conv, hfun⟩
  rw [StrongConvexOn, UniformConvexOn]
  constructor
  · exact conv
  · intro y yin z zin a b anneg bnneg absum1; simp
    rw [mul_add, add_assoc, add_comm (a * (‖y - x‖ ^ 2 / 2)), ← add_assoc]
    rw [mul_add, ← add_assoc, ← add_sub _ (a * (‖y - x‖ ^ 2 / 2)), add_assoc]
    apply add_le_add
    · rw [← smul_eq_mul, ← smul_eq_mul]
      apply hfun yin zin anneg bnneg absum1
    · field_simp; rw [div_le_div_iff_of_pos_right, add_sub]
      have eq1 : a • y + b • z - x = a • (y - x) + b • (z - x) := by
        rw [smul_sub, smul_sub, add_comm_sub, sub_sub, ← add_smul, add_comm b a]
        rw [absum1, one_smul, ← add_sub]
      have eq2 (u v : E) : ‖a • u + b • v‖ ^ 2 = b * ‖v‖ ^ 2
        + a * ‖u‖ ^ 2 - a * b * ‖u - v‖ ^ 2 := by
        rw [norm_add_sq_real, norm_sub_sq_real]
        rw [inner_smul_left, inner_smul_right, norm_smul, norm_smul]; field_simp
        rw [add_comm (b * ‖v‖ ^ 2), mul_pow, sq_abs, mul_pow, sq_abs]
        rw [mul_add, ← sub_sub, mul_sub, ← sub_add]
        rw [add_sub_right_comm, add_sub_right_comm, ← sub_mul, ← add_sub, ← sub_mul]
        nth_rw 3 [← mul_one a]; rw [← absum1, mul_add]
        nth_rw 5 [← mul_one b]; rw [← absum1, mul_add, mul_comm b a]
        rw [pow_two, pow_two b]; simp; rw [add_right_comm, add_left_cancel_iff]
        rw [mul_mul_mul_comm, mul_comm a 2, mul_assoc]
      have eq3 : y - z = (y - x) - (z - x) := by simp
      have eq4 (u v : E) : ‖a • u + b • v‖ ^ 2 ≤ b * ‖v‖ ^ 2
        + a * ‖u‖ ^ 2 - a * b * ‖u - v‖ ^ 2 := by rw [eq2]
      let u := y - x
      let v := z - x
      rw [eq1, eq3];
      show ‖a • u + b • v‖ ^ 2 ≤ b * ‖v‖ ^ 2 + a * ‖u‖ ^ 2 - a * b * ‖u - v‖ ^ 2
      apply eq4 u v
      simp

end Strongly_Convex

section

variable [CompleteSpace E]

theorem Strong_Convex_lower (hsc : StrongConvexOn s m f) (hf : ∀ x ∈ s, HasGradientAt f (f' x) x) :
    ∀ x ∈ s, ∀ y ∈ s, inner (f' x - f' y) (x - y) ≥ m * ‖x - y‖ ^ 2 := by
  intro x xs y ys
  have cvx := strongConvexOn_iff_convex.mp hsc
  have grd := sub_normsquare_gradient hf m
  have grm := Convex_monotone_gradient' cvx grd x xs y ys
  rw [sub_sub, add_sub, add_comm, ← add_sub, ← sub_sub, inner_sub_left, ← smul_sub] at grm
  apply le_of_sub_nonneg at grm
  rw [real_inner_smul_left, real_inner_self_eq_norm_sq] at grm
  apply grm

theorem Lower_Strong_Convex (hf : ∀ x ∈ s, HasGradientAt f (f' x) x) (hs : Convex ℝ s)
    (h : ∀ x ∈ s, ∀ y ∈ s, inner (f' x - f' y) (x - y) ≥ m * ‖x - y‖ ^ 2) :
    StrongConvexOn s m f := by
  apply strongConvexOn_iff_convex.mpr
  have grd := sub_normsquare_gradient hf m
  apply monotone_gradient_convex' hs grd
  intro x xs y ys
  specialize h x xs y ys
  rw [sub_sub, add_sub, add_comm, ← add_sub, ← sub_sub, inner_sub_left, ← smul_sub]
  apply sub_nonneg_of_le
  rw [real_inner_smul_left, real_inner_self_eq_norm_sq]
  apply h

theorem Strong_Convex_iff_lower (hf : ∀ x ∈ s, HasGradientAt f (f' x) x) (hs : Convex ℝ s) :
    StrongConvexOn s m f ↔ ∀ x ∈ s, ∀ y ∈ s, inner (f' x - f' y) (x - y) ≥ m * ‖x - y‖ ^ 2 :=
  ⟨fun hsc x xs y ys ↦ Strong_Convex_lower hsc hf x xs y ys, fun h ↦ Lower_Strong_Convex hf hs h⟩

theorem Strong_Convex_second_lower (hsc: StrongConvexOn s m f)
    (hf : ∀ x ∈ s, HasGradientAt f (f' x) x) : ∀ x ∈ s, ∀ y ∈ s,
    f y ≥ f x + inner (f' x) (y - x) + m / 2 * ‖y - x‖ ^ 2 := by
  intro x xs y ys
  have cvx := strongConvexOn_iff_convex.mp hsc
  have grd := sub_normsquare_gradient hf m x xs
  let g := fun x ↦ f' x - m • x
  have : g x = f' x - m • x := by rfl
  rw [← this] at grd
  have foc := Convex_first_order_condition' grd cvx xs y ys
  rw [this] at foc
  apply sub_nonneg_of_le at foc
  apply le_of_sub_nonneg
  rw [sub_right_comm, sub_add, ← sub_add, add_sub, add_sub_right_comm] at foc
  rw [inner_sub_left, sub_add, sub_sub (f y), sub_sub, add_sub, add_sub] at foc
  rw [← real_inner_self_eq_norm_sq, ← real_inner_self_eq_norm_sq, sub_add, ← mul_sub] at foc
  nth_rw 7 [← zero_add x] at foc
  nth_rw 5 [← sub_zero y] at foc
  nth_rw 1 [← sub_self y] at foc
  rw [← sub_self x] at foc
  rw [sub_add, ← sub_add y x x, add_comm (y - x), inner_sub_right x, inner_add_right y] at foc
  rw [real_inner_comm x y, sub_right_comm (inner x y), ← sub_sub, sub_self, sub_sub 0] at foc
  rw [← inner_add_left, zero_sub, mul_neg, sub_neg_eq_add] at foc
  have : m = m / 2 * 2 := by simp
  nth_rw 1 [this] at foc
  rw [← smul_smul, inner_smul_left, two_smul] at foc
  simp [map_div₀, map_ofNat] at foc
  rw [sub_add, ← mul_sub, ← inner_sub_left, ← sub_sub, sub_right_comm,← add_sub x, add_sub_cancel] at foc
  have : x - y = - (y - x) := by simp
  rw [this, inner_neg_left, mul_neg, sub_neg_eq_add, real_inner_self_eq_norm_sq] at foc
  linarith

end
