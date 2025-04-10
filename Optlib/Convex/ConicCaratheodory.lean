/-
Copyright (c) 2024 Shengyang Xu. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Shengyang Xu
-/
import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# ConicCaratheodory

## Main results

This file contains the proof of conic version of Caratheodory theorem.

-/

open Finset

variable {n : ℕ} {s : Finset ℕ} {V : ℕ → (EuclideanSpace ℝ (Fin n))}
variable {x : EuclideanSpace ℝ (Fin n)}

/- Nonnegative Coefficients -/
def quadrant : Set (ℕ → ℝ) := {x : ℕ → ℝ | ∀ i : ℕ, 0 ≤ x i}

/- Restrict domain to Finset ℕ -/
def coe (s: Finset ℕ) : s → ℕ := fun i ↦ i

/- Definition of cone with finite base -/
def cone (s : Finset ℕ) (V : ℕ → (EuclideanSpace ℝ (Fin n))) : Set (EuclideanSpace ℝ (Fin n)) :=
  (fun x ↦ Finset.sum s (fun i => x i • V i)) '' quadrant

/- An Alternative Proof of conic version of Caratheodory Theorem.
   Reference : https://courses.engr.illinois.edu/cs598csc/sp2010/Lectures/Lecture2.pdf Theorem 18.
   This lemma states that : If x is in the cone generated by Finset s whose elements are not
   linear independent, then it is in the cone generated by a strict subset of s -/
private lemma mem_conic_erase (s : Finset ℕ) (V : ℕ → (EuclideanSpace ℝ (Fin n)))
    (nidp : ¬LinearIndependent ℝ (V ∘ coe s)) (xincV : x ∈ cone s V) :
    ∃ y : s, x ∈ cone (s.erase y) V := by
  simp [cone, quadrant] at xincV
  rcases xincV with ⟨t, tin, xdecompose⟩
  by_cases ht : ∃ i₀ : s, t i₀ = 0
  · rcases ht with ⟨i₀, hti₀⟩
    use i₀; simp [cone]; use t
    constructor
    · simp [quadrant, tin]
    · rw [← xdecompose, hti₀]; simp
  · have tpos : ∀ i : s, 0 < t i := by
      intro i; simp at ht; exact Ne.lt_of_le' (ht i i.prop) (tin i)
    let V₀ := fun i : ℕ ↦ t i • V i
    have nidp' : ¬LinearIndependent ℝ (V₀ ∘ coe s) := by
      obtain ⟨τ, k, ksum0, kextn0⟩ := not_linearIndependent_iff.1 nidp
      let k' := fun i : s ↦ k i / t i
      rw [not_linearIndependent_iff]
      use τ; use k'; constructor
      · rw [← ksum0]; simp [k', V₀, coe]
        apply Finset.sum_congr; simp
        intro i _; rw [smul_smul, div_mul_cancel₀]; linarith [tpos i]
      · obtain ⟨i₀, i₀in, ki₀ne0⟩ := kextn0
        use i₀ ; use i₀in
        simp [k']; push_neg; use ki₀ne0; linarith [tpos i₀]
    have pos_case (k : s → ℝ) (ksum0 : (Finset.sum univ fun i ↦ k i • (V₀ ∘ coe s) i) = 0)
        (j : s) (kjpos : 0 < k j) : ∃ y : s, x ∈ cone (s.erase y) V := by
      have maxk : ∃ j₀ : s, ∀ j : s, k j ≤ k j₀ := by
        let k' := fun i : ℕ ↦ if hi : i ∈ s then k ⟨i, hi⟩ else 0
        have sne : s.Nonempty := by use j; simp
        apply (s.exists_max_image k') at sne
        obtain ⟨j₀, j₀in, hj₀⟩ := sne; use ⟨j₀, j₀in⟩
        intro j; specialize hj₀ j j.prop; simp [k', j₀in] at hj₀; apply hj₀
      rcases maxk with ⟨j₀, kj₀max⟩
      have kj₀pos : 0 < k j₀ := by linarith [kj₀max j]
      let β := fun i : ℕ => if hi : i ∈ s then (t i * k ⟨i, hi⟩) else 0
      let α := fun i : ℕ => t i - β i / k j₀
      have αpos : ∀ i : s, 0 ≤ α i := by
        intro i; simp [α, β]; rw [mul_div_assoc]
        apply mul_le_of_le_one_right; linarith [tpos i]
        rw [div_le_one]; linarith [kj₀max i]; linarith
      use j₀; simp [cone]; use α
      constructor
      · simp [quadrant]; intro i
        by_cases hi : i ∈ s
        · linarith [αpos ⟨i, hi⟩]
        · simp [α, hi, β]; linarith [tin i]
      · have hαj₀ : α j₀ = 0 := by field_simp [α, β]
        rw [hαj₀, ← xdecompose]; simp [α]
        have aux : (Finset.sum s fun x ↦ (t x - β x / k j₀) • V x) =
            (Finset.sum s fun x ↦ t x • V x) - (1 / k j₀) • (Finset.sum s fun x ↦ β x • V x) := by
          rw [Finset.smul_sum, ← Finset.sum_sub_distrib]; congr
          ext i j; rw [smul_smul, sub_smul]; field_simp
        rw [aux]; simp; right
        simp [V₀, coe] at ksum0; rw [← ksum0, ← Finset.sum_attach]
        congr; ext x l
        simp [β]; rw [← mul_assoc, mul_comm (t x) (k x)]
    obtain ⟨k, ksum0, kextn0⟩ := Fintype.not_linearIndependent_iff.1 nidp'
    rcases kextn0 with ⟨j, kjneq0⟩
    by_cases kjpos: 0 < k j
    · apply pos_case k ksum0 j kjpos
    · let k' := fun i : s => - k i
      have k'sum0 : (Finset.sum univ fun i ↦ k' i • (V₀ ∘ coe s) i) = 0 := by
        simp at ksum0; simp [k', ksum0]
      have k'jpos : 0 < k' j := by simp [k']; simp at kjpos; exact Ne.lt_of_le kjneq0 kjpos
      apply pos_case k' k'sum0 j k'jpos

theorem conic_Caratheodory (s : Finset ℕ) (V : ℕ → (EuclideanSpace ℝ (Fin n))):
    ∀ x ∈ cone s V, ∃ τ : Finset ℕ,
    (τ ⊆ s) ∧ (x ∈ cone τ V) ∧ (LinearIndependent ℝ (V ∘ coe τ)) ∧
    (∀ σ : Finset ℕ, σ ⊆ s → x ∈ cone σ V → τ.card ≤ σ.card) := by
  intro x xin
  let idx := {τ : Finset ℕ | (τ ⊆ s) ∧ (x ∈ cone τ V)}
  have finite_idx : idx.Finite := by
    have finite_ps : Finite s.powerset := by apply Finset.finite_toSet
    have idx_sub_ps : idx ⊆ s.powerset := by
      intro τ τin
      simp [idx] at τin
      apply Finset.mem_powerset.2 τin.1
    apply Set.Finite.subset _ idx_sub_ps
    apply finite_ps
  have ne_idx : Set.Nonempty idx := by use s; simp [idx, xin]
  let to_card : Finset ℕ → ℝ := fun x => x.card
  obtain ⟨τ, τin, τcardmin⟩ := idx.exists_min_image to_card finite_idx ne_idx
  use τ; simp [idx] at τin; use τin.1; use τin.2; constructor
  · by_contra nidp
    obtain ⟨y, xinerase⟩ := mem_conic_erase τ V nidp τin.2
    let τ' := erase τ y
    specialize τcardmin τ'; simp [idx, τ'] at τcardmin
    have τ'subs : τ' ⊆ s := by
      simp [τ']; apply subset_trans _ τin.1
      apply Finset.erase_subset
    specialize τcardmin τ'subs xinerase
    simp [to_card] at τcardmin
    by_cases τempty : τ.Nonempty
    · absurd τcardmin; simp; exact τempty
    · have coneτVempty : cone τ V = ∅ := by
        simp [not_nonempty_iff_eq_empty] at τempty
        rw [τempty] at τin
        subst τempty
        simp only [empty_subset, true_and] at τin
        simp only [linearIndependent_subsingleton_index_iff, Function.comp_apply] at nidp
        simp only [ne_eq, IsEmpty.forall_iff, not_true_eq_false] at nidp
      have idxempty : idx = ∅ := by
        simp only [Set.mem_empty_iff_false, coneτVempty] at τin
        simp only [and_false] at τin
      have f : False := by
        rw [idxempty] at ne_idx
        simp only [Set.finite_empty, Set.not_nonempty_empty, idx] at ne_idx
      exact f
  · intro σ σsubs; specialize τcardmin σ
    simp [idx, to_card] at τcardmin
    apply τcardmin σsubs
