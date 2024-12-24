import Convex

open Set Finset


theorem extracted_0 {E : Type u_1} {f : E → ℝ} {g : ℕ → E} (mono : ∀ (k : ℕ), f (g (k + 1)) ≤ f (g k)) (n : ℕ) :
  ∑ k ∈ Finset.range (n + 1), f (g (k + 1)) ≥ (↑n + 1) * f (g (n + 2)) := sorry


theorem extracted_1 {E : Type u_1} {f : E → ℝ} {g : ℕ → E} (q : ℕ)
  (IH1 : ∑ k ∈ Finset.range (q + 1), f (g (k + 1)) ≥ (↑q + 1) * f (g (q + 2)))
  (mono : f (g (q + 2 + 1)) ≤ f (g (q + 2))) :
  ∑ k ∈ Finset.range (q.succ + 1), f (g (k + 1)) = ∑ k ∈ Finset.range (q + 1), f (g (k + 1)) + f (g (q + 2)) := sorry


theorem extracted_2 {E : Type u_1} {f : E → ℝ} {g : ℕ → E} (q : ℕ)
  (IH1 : ∑ k ∈ Finset.range (q + 1), f (g (k + 1)) ≥ (↑q + 1) * f (g (q + 2)))
  (mono : f (g (q + 2 + 1)) ≤ f (g (q + 2))) :
  ∑ k ∈ Finset.range (q + 1), f (g (k + 1)) + f (g (q + 2)) ≥ (↑q + 1) * f (g (q + 2)) + f (g (q + 2)) := sorry


theorem extracted_3 {E : Type u_1} {f : E → ℝ} {g : ℕ → E} (q : ℕ)
  (IH1 : ∑ k ∈ Finset.range (q + 1), f (g (k + 1)) ≥ (↑q + 1) * f (g (q + 2)))
  (mono : f (g (q + 2 + 1)) ≤ f (g (q + 2))) :
  (↑q + 1) * f (g (q + 2)) + f (g (q + 2)) = (↑q + 2) * f (g (q + 2)) := sorry


theorem extracted_4 {E : Type u_1} {f : E → ℝ} {g : ℕ → E} (q : ℕ)
  (IH1 : ∑ k ∈ Finset.range (q + 1), f (g (k + 1)) ≥ (↑q + 1) * f (g (q + 2)))
  (mono : f (g (q + 2 + 1)) ≤ f (g (q + 2))) : 0 ≤ ↑q + 2 := sorry


theorem extracted_5 {E : Type u_1} {f : E → ℝ} {g : ℕ → E} (q : ℕ)
  (IH1 : ∑ k ∈ Finset.range (q + 1), f (g (k + 1)) ≥ (↑q + 1) * f (g (q + 2)))
  (mono : f (g (q + 2 + 1)) ≤ f (g (q + 2))) : (↑q + 2) * f (g (q + 3)) = (↑q.succ + 1) * f (g (q.succ + 2)) := sorry


theorem extracted_6 {E : Type u_1} {f : E → ℝ} {g : ℕ → E} (mono : ∀ (k : ℕ), f (g (k + 1)) ≤ f (g k)) (n : ℕ) :
  (∑ k ∈ Finset.range (n.succ + 1), f (g (k + 1))) / (↑n.succ + 1) ≥ f (g (n + 2)) := sorry


theorem extracted_7 {E : Type u_1} {f : E → ℝ} {g : ℕ → E} (mono : ∀ (k : ℕ), f (g (k + 1)) ≤ f (g k)) (n : ℕ)
  (h : (↑n + 1) * f (g (n.succ + 1)) / (↑n + 1 + 1) ≤ (∑ k ∈ Finset.range n.succ, f (g (k + 1))) / (↑n + 1 + 1)) :
  (∑ k ∈ Finset.range (n.succ + 1), f (g (k + 1))) / (↑n.succ + 1) =
    (∑ k ∈ Finset.range n.succ, f (g (k + 1))) / (↑n.succ + 1) + f (g (n.succ + 1)) / (↑n.succ + 1) := sorry


theorem extracted_8 {E : Type u_1} {f : E → ℝ} {g : ℕ → E} (mono : ∀ (k : ℕ), f (g (k + 1)) ≤ f (g k)) (n : ℕ)
  (h : (↑n + 1) * f (g (n.succ + 1)) / (↑n + 1 + 1) ≤ (∑ k ∈ Finset.range n.succ, f (g (k + 1))) / (↑n + 1 + 1)) :
  (∑ k ∈ Finset.range n.succ, f (g (k + 1))) / (↑n.succ + 1) + f (g (n.succ + 1)) / (↑n.succ + 1) ≥
    ↑n.succ * f (g (n.succ + 1)) / (↑n.succ + 1) + f (g (n.succ + 1)) / (↑n.succ + 1) := sorry


theorem extracted_9 {E : Type u_1} {f : E → ℝ} {g : ℕ → E} (mono : ∀ (k : ℕ), f (g (k + 1)) ≤ f (g k)) (n : ℕ)
  (h : (↑n + 1) * f (g (n.succ + 1)) / (↑n + 1 + 1) ≤ (∑ k ∈ Finset.range n.succ, f (g (k + 1))) / (↑n + 1 + 1)) :
  ↑n.succ * f (g (n.succ + 1)) / (↑n.succ + 1) + f (g (n.succ + 1)) / (↑n.succ + 1) = f (g (n + 2)) := sorry


theorem extracted_10 {E : Type u_1} {xm : E} {f : E → ℝ} {g : ℕ → E} (mono : ∀ (k : ℕ), f (g (k + 1)) ≤ f (g k))
  (n : ℕ) : f (g (n + 1)) - f xm ≤ (∑ k ∈ Finset.range (n + 1), (f (g (k + 1)) - f xm)) / (↑n + 1) := sorry


theorem extracted_11 {E : Type u_1} {xm : E} {f : E → ℝ} {g : ℕ → E} (mono : ∀ (k : ℕ), f (g (k + 1)) ≤ f (g k))
  (j : ℕ) (a : f (g (j + 1)) - f xm ≤ (∑ k ∈ Finset.range (j + 1), (f (g (k + 1)) - f xm)) / (↑j + 1)) :
  f (g (j + 2)) ≤ (∑ k ∈ Finset.range (j.succ + 1), f (g (k + 1))) / (↑j.succ + 1) := sorry


theorem extracted_12 {E : Type u_1} {xm : E} {f : E → ℝ} {g : ℕ → E} (mono : ∀ (k : ℕ), f (g (k + 1)) ≤ f (g k))
  (j : ℕ) (a : f (g (j + 1)) - f xm ≤ (∑ k ∈ Finset.range (j + 1), (f (g (k + 1)) - f xm)) / (↑j + 1)) :
  (∑ k ∈ Finset.range (j.succ + 1), f (g (k + 1))) / (↑j.succ + 1) =
    (∑ k ∈ Finset.range (j.succ + 1), f (g (k + 1))) / (↑j + 2) - f xm * 1 + f xm := sorry


theorem extracted_13 {E : Type u_1} {xm : E} {f : E → ℝ} {g : ℕ → E} (mono : ∀ (k : ℕ), f (g (k + 1)) ≤ f (g k))
  (j : ℕ) (a : f (g (j + 1)) - f xm ≤ (∑ k ∈ Finset.range (j + 1), (f (g (k + 1)) - f xm)) / (↑j + 1)) :
  (∑ k ∈ Finset.range (j.succ + 1), f (g (k + 1))) / (↑j + 2) - f xm * 1 + f xm =
    (∑ k ∈ Finset.range (j.succ + 1), f (g (k + 1))) / (↑j + 2) - f xm * ((↑j + 2) / (↑j + 2)) + f xm := sorry


theorem extracted_14 {E : Type u_1} {xm : E} {f : E → ℝ} {g : ℕ → E} (mono : ∀ (k : ℕ), f (g (k + 1)) ≤ f (g k))
  (j : ℕ) (a : f (g (j + 1)) - f xm ≤ (∑ k ∈ Finset.range (j + 1), (f (g (k + 1)) - f xm)) / (↑j + 1)) :
  (∑ k ∈ Finset.range (j.succ + 1), f (g (k + 1))) / (↑j + 2) - f xm * ((↑j + 2) / (↑j + 2)) + f xm =
    (∑ k ∈ Finset.range (j.succ + 1), f (g (k + 1)) - (↑j + 1 + 1) * f xm) / (↑j + 1 + 1) + f xm := sorry


theorem extracted_15 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {x₀ : E} [p : Gradient_Descent_fix_stepsize f f' x₀] (k : ℕ) :
  (fun x => Gradient_Descent_fix_stepsize.a f f' x₀) k > 0 := sorry


theorem extracted_16 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (hfun : ConvexOn ℝ univ f) (x y : E) : f x ≤ f y + ⟪f' x, x - y⟫_ℝ := sorry


theorem extracted_17 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (hfun : ConvexOn ℝ univ f) (x : E) : x ∈ univ := sorry


theorem extracted_18 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (hfun : ConvexOn ℝ univ f) (y : E) : y ∈ univ := sorry


theorem extracted_19 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {l : NNReal} {a : ℝ} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (h₂ : ↑l > 0) (ha₁ : ↑l ≤ 1 / a) (ha₂ : a > 0) (h₃ : LipschitzWith l f') (x : E) :
  f (x - a • f' x) ≤ f x - a / 2 * ‖f' x‖ ^ 2 := sorry


theorem extracted_20 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {l : NNReal} {a : ℝ} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (h₂ : ↑l > 0) (ha₁ : ↑l ≤ 1 / a) (ha₂ : a > 0) (h₃ : LipschitzWith l f') (x : E) :
  f x + ⟪f' x, x - a • f' x - x⟫_ℝ + ↑l / 2 * ‖x - a • f' x - x‖ ^ 2 = f x + (↑l / 2 * a * a - a) * ‖f' x‖ ^ 2 := sorry


theorem extracted_21 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {l : NNReal} {a : ℝ} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (h₂ : ↑l > 0) (ha₁ : ↑l ≤ 1 / a) (ha₂ : a > 0) (h₃ : LipschitzWith l f') (x : E) :
  f x + (↑l / 2 * a * a - a) * ‖f' x‖ ^ 2 ≤ f x + -a / 2 * ‖f' x‖ ^ 2 := sorry


theorem extracted_22 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {l : NNReal} {a : ℝ} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (h₂ : ↑l > 0) (ha₁ : ↑l ≤ 1 / a) (ha₂ : a > 0) (h₃ : LipschitzWith l f') : ↑l / 2 * a * a = ↑l * a * (a / 2) := sorry


theorem extracted_23 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {l : NNReal} {a : ℝ} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (h₂ : ↑l > 0) (ha₁ : ↑l ≤ 1 / a) (ha₂ : a > 0) (h₃ : LipschitzWith l f') : ↑l * a * (a / 2) ≤ 1 * (a / 2) := sorry


theorem extracted_24 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {l : NNReal} {a : ℝ} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (h₂ : ↑l > 0) (ha₁ : ↑l ≤ 1 / a) (ha₂ : a > 0) (h₃ : LipschitzWith l f') : a / 2 ≤ a / 2 := sorry


theorem extracted_25 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {l : NNReal} {a : ℝ} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (h₂ : ↑l > 0) (ha₁ : ↑l ≤ 1 / a) (ha₂ : a > 0) (h₃ : LipschitzWith l f') : 0 ≤ ↑l * a := sorry


theorem extracted_26 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {l : NNReal} {a : ℝ} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (h₂ : ↑l > 0) (ha₁ : ↑l ≤ 1 / a) (ha₂ : a > 0) (h₃ : LipschitzWith l f') : 0 ≤ a / 2 := sorry


theorem extracted_27 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {l : NNReal} {a : ℝ} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (h₂ : ↑l > 0) (ha₁ : ↑l ≤ 1 / a) (ha₂ : a > 0) (h₃ : LipschitzWith l f') : 1 * (a / 2) = -a / 2 + a := sorry


theorem extracted_28 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {l : NNReal} {a : ℝ} (h₁ : ∀ (x₁ : E), HasGradientAt f (f' x₁) x₁)
  (h₂ : ↑l > 0) (ha₁ : ↑l ≤ 1 / a) (ha₂ : a > 0) (h₃ : LipschitzWith l f') (x : E) :
  f x + -a / 2 * ‖f' x‖ ^ 2 = f x - a / 2 * ‖f' x‖ ^ 2 := sorry


theorem extracted_29 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂ : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀)) (k : ℕ) :
  f (Gradient_Descent_fix_stepsize.x f f' x₀ (k + 1)) ≤
    f xm +
      1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) *
        (‖Gradient_Descent_fix_stepsize.x f f' x₀ k - xm‖ ^ 2 -
          ‖Gradient_Descent_fix_stepsize.x f f' x₀ (k + 1) - xm‖ ^ 2) := sorry


theorem extracted_30 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂ : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀)) :
  ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀ := sorry


theorem extracted_31 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂_1 : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀))
  (step₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) (x : E) :
  f (x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x) ≤
    f xm +
      1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) *
        (‖x - xm‖ ^ 2 - ‖x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x - xm‖ ^ 2) := sorry


theorem extracted_32 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂_1 : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀))
  (step₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) :
  1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * (2 * Gradient_Descent_fix_stepsize.a f f' x₀) = 1 := sorry


theorem extracted_33 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂_1 : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀))
  (step₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) : Gradient_Descent_fix_stepsize.a f f' x₀ ≠ 0 := sorry


theorem extracted_34 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂_1 : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀))
  (step₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) (x : E)
  (t1 : 1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * (2 * Gradient_Descent_fix_stepsize.a f f' x₀) = 1) :
  ⟪f' x, x - xm⟫_ℝ - Gradient_Descent_fix_stepsize.a f f' x₀ / 2 * ‖f' x‖ ^ 2 =
    1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) *
      (‖x - xm‖ ^ 2 - ‖x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x - xm‖ ^ 2) := sorry


theorem extracted_35 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂_1 : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀))
  (step₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) (x : E)
  (t1 : 1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * (2 * Gradient_Descent_fix_stepsize.a f f' x₀) = 1) :
  ‖x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x - xm‖ ^ 2 =
    ‖x - xm‖ ^ 2 - 2 * Gradient_Descent_fix_stepsize.a f f' x₀ * ⟪f' x, x - xm⟫_ℝ +
      ‖Gradient_Descent_fix_stepsize.a f f' x₀ • f' x‖ ^ 2 := sorry


theorem extracted_36 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂_1 : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀))
  (step₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) (x : E)
  (t1 : 1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * (2 * Gradient_Descent_fix_stepsize.a f f' x₀) = 1)
  (t2₁ :
    ‖x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x - xm‖ ^ 2 =
      ‖x - xm‖ ^ 2 - 2 * Gradient_Descent_fix_stepsize.a f f' x₀ * ⟪f' x, x - xm⟫_ℝ +
        ‖Gradient_Descent_fix_stepsize.a f f' x₀ • f' x‖ ^ 2) :
  1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) *
      (‖x - xm‖ ^ 2 - ‖x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x - xm‖ ^ 2) =
    1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * (2 * Gradient_Descent_fix_stepsize.a f f' x₀) *
        ⟪f' x, x - xm⟫_ℝ +
      1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * -‖Gradient_Descent_fix_stepsize.a f f' x₀ • f' x‖ ^ 2 := sorry


theorem extracted_37 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂_1 : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀))
  (step₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) (x : E)
  (t1 : 1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * (2 * Gradient_Descent_fix_stepsize.a f f' x₀) = 1)
  (t2₁ :
    ‖x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x - xm‖ ^ 2 =
      ‖x - xm‖ ^ 2 - 2 * Gradient_Descent_fix_stepsize.a f f' x₀ * ⟪f' x, x - xm⟫_ℝ +
        ‖Gradient_Descent_fix_stepsize.a f f' x₀ • f' x‖ ^ 2) :
  1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * ⟪f' x, x - xm⟫_ℝ +
      1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * -‖Gradient_Descent_fix_stepsize.a f f' x₀ • f' x‖ ^ 2 =
    ⟪f' x, x - xm⟫_ℝ +
      1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * -‖Gradient_Descent_fix_stepsize.a f f' x₀ • f' x‖ ^ 2 := sorry


theorem extracted_38 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂_1 : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀))
  (step₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) (x : E)
  (t1 : 1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * (2 * Gradient_Descent_fix_stepsize.a f f' x₀) = 1)
  (t2₁ :
    ‖x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x - xm‖ ^ 2 =
      ‖x - xm‖ ^ 2 - 2 * Gradient_Descent_fix_stepsize.a f f' x₀ * ⟪f' x, x - xm⟫_ℝ +
        ‖Gradient_Descent_fix_stepsize.a f f' x₀ • f' x‖ ^ 2) :
  ⟪f' x, x - xm⟫_ℝ +
      1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * -‖Gradient_Descent_fix_stepsize.a f f' x₀ • f' x‖ ^ 2 =
    ⟪f' x, x - xm⟫_ℝ -
      1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) *
          (Gradient_Descent_fix_stepsize.a f f' x₀ * Gradient_Descent_fix_stepsize.a f f' x₀) *
        ‖f' x‖ ^ 2 := sorry


theorem extracted_39 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂_1 : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀))
  (step₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) (x : E)
  (t1 : 1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * (2 * Gradient_Descent_fix_stepsize.a f f' x₀) = 1)
  (t2₁ :
    ‖x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x - xm‖ ^ 2 =
      ‖x - xm‖ ^ 2 - 2 * Gradient_Descent_fix_stepsize.a f f' x₀ * ⟪f' x, x - xm⟫_ℝ +
        ‖Gradient_Descent_fix_stepsize.a f f' x₀ • f' x‖ ^ 2) :
  ⟪f' x, x - xm⟫_ℝ -
      1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) *
          (Gradient_Descent_fix_stepsize.a f f' x₀ * Gradient_Descent_fix_stepsize.a f f' x₀) *
        ‖f' x‖ ^ 2 =
    ⟪f' x, x - xm⟫_ℝ - Gradient_Descent_fix_stepsize.a f f' x₀ / 2 * ‖f' x‖ ^ 2 := sorry


theorem extracted_40 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂_1 : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀))
  (step₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) (x : E)
  (t1 : 1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * (2 * Gradient_Descent_fix_stepsize.a f f' x₀) = 1)
  (t2 :
    ⟪f' x, x - xm⟫_ℝ - Gradient_Descent_fix_stepsize.a f f' x₀ / 2 * ‖f' x‖ ^ 2 =
      1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) *
        (‖x - xm‖ ^ 2 - ‖x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x - xm‖ ^ 2)) :
  f (x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x) ≤
    f x - Gradient_Descent_fix_stepsize.a f f' x₀ / 2 * ‖f' x‖ ^ 2 := sorry


theorem extracted_41 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂_1 : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀))
  (step₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) (x : E)
  (t1 : 1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * (2 * Gradient_Descent_fix_stepsize.a f f' x₀) = 1)
  (t2 :
    ⟪f' x, x - xm⟫_ℝ - Gradient_Descent_fix_stepsize.a f f' x₀ / 2 * ‖f' x‖ ^ 2 =
      1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) *
        (‖x - xm‖ ^ 2 - ‖x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x - xm‖ ^ 2)) :
  f x - Gradient_Descent_fix_stepsize.a f f' x₀ / 2 * ‖f' x‖ ^ 2 ≤
    f xm + ⟪f' x, x - xm⟫_ℝ - Gradient_Descent_fix_stepsize.a f f' x₀ / 2 * ‖f' x‖ ^ 2 := sorry


theorem extracted_42 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂_1 : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀))
  (step₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) (x : E)
  (t1 : 1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) * (2 * Gradient_Descent_fix_stepsize.a f f' x₀) = 1)
  (t2 :
    ⟪f' x, x - xm⟫_ℝ - Gradient_Descent_fix_stepsize.a f f' x₀ / 2 * ‖f' x‖ ^ 2 =
      1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) *
        (‖x - xm‖ ^ 2 - ‖x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x - xm‖ ^ 2)) :
  f xm + ⟪f' x, x - xm⟫_ℝ - Gradient_Descent_fix_stepsize.a f f' x₀ / 2 * ‖f' x‖ ^ 2 =
    f xm +
      1 / (2 * Gradient_Descent_fix_stepsize.a f f' x₀) *
        (‖x - xm‖ ^ 2 - ‖x - Gradient_Descent_fix_stepsize.a f f' x₀ • f' x - xm‖ ^ 2) := sorry


theorem extracted_43 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂ : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀)) (k : ℕ) :
  f (Gradient_Descent_fix_stepsize.x f f' x₀ (k + 1)) - f xm ≤
    1 / (2 * (↑k + 1) * Gradient_Descent_fix_stepsize.a f f' x₀) * ‖x₀ - xm‖ ^ 2 := sorry


theorem extracted_44 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂ : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀)) :
  ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀ := sorry


theorem extracted_45 {E : Type u_1} [inst : NormedAddCommGroup E] [inst_1 : InnerProductSpace ℝ E]
  [inst_2 : CompleteSpace E] {f : E → ℝ} {f' : E → E} {xm x₀ : E} {alg : Gradient_Descent_fix_stepsize f f' x₀}
  (hfun : ConvexOn ℝ univ f)
  (step₂ : Gradient_Descent_fix_stepsize.a f f' x₀ ≤ 1 / ↑(Gradient_Descent_fix_stepsize.l f f' x₀)) (k : ℕ)
  (step1₂ : ↑(Gradient_Descent_fix_stepsize.l f f' x₀) ≤ 1 / Gradient_Descent_fix_stepsize.a f f' x₀)
  (this_1 : LipschitzWith (Gradient_Descent_fix_stepsize.l f f' x₀) f')
  (this : Gradient_Descent_fix_stepsize.l f f' x₀ > 0) :
  1 / (2 * (↑k + 1) * Gradient_Descent_fix_stepsize.a f f' x₀) > 0 := sorry
