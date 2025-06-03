(define (domain phase-shift)
  (:requirements :typing :negative-preconditions :conditional-effects)
  (:types
      coord                         ; board squares (phase–independent id)
      phase                         ; alpha / beta
  )

  (:constants alpha beta - phase)

  (:predicates
      ;; world anatomy ----------------------------------------------------
      (adj-north ?c1 ?c2 - coord)   ; ?c2 is one row north  of ?c1
      (adj-south ?c1 ?c2 - coord)   ; ?c2 is one row south  of ?c1
      (adj-west  ?c1 ?c2 - coord)   ; ?c2 is one column west of ?c1
      (adj-east  ?c1 ?c2 - coord)   ; ?c2 is one column east of ?c1

      (walkable ?c - coord ?p - phase)       ; floor / beacon / exit / key
      (rift     ?c - coord ?p - phase)       ; lethal in that phase
      (gate     ?c - coord ?p - phase)       ; closed gate in that phase
      (beacon   ?c - coord)                  ; square is a beacon
      (exit-tile ?c - coord)                 ; square is an exit
      (key-tile  ?c - coord)                 ; square contains the key

      ;; dynamic state ----------------------------------------------------
      (player-at ?c - coord)                 ; current player location
      (phase-current ?p - phase)             ; alpha OR beta is active
      (beacon-on ?c - coord)                 ; beacon already activated
      (gate-open ?c - coord)                 ; gate is unlocked (both phases)
      (have-key)                             ; player has collected key
  )

  ;; ──────────────────────────────────────────────────────────────────
  ;; MOVEMENT – four cardinal directions (identical skeleton)
  ;; ──────────────────────────────────────────────────────────────────
  (:action move-north
    :parameters (?from ?to - coord ?ph - phase)
    :precondition (and
        (player-at ?from)            (phase-current ?ph)
        (adj-north ?from ?to)
        (walkable ?to ?ph)
        (not (gate ?to ?ph))         ; closed gates block
        (not (rift ?to ?ph))
    )
    :effect (and
        (not (player-at ?from))
        (player-at ?to)

        ;; collect key ----------------------------------------------------
        (when (key-tile ?to) (have-key))

        ;; open gate tiles globally ---------------------------------------
        (when (and (key-tile ?to))        ; triggered when key just taken
              (forall (?c - coord)
                  (when (gate ?c alpha) (gate-open ?c))
              )
        )

        ;; beacon activation ----------------------------------------------
        (when (beacon ?to) (beacon-on ?to))
    )
  )

  ;; south ------------------------------------------------------------
  (:action move-south
    :parameters (?from ?to - coord ?ph - phase)
    :precondition (and
        (player-at ?from) (phase-current ?ph)
        (adj-south ?from ?to)
        (walkable ?to ?ph)
        (not (gate ?to ?ph))
        (not (rift ?to ?ph))
    )
    :effect (and
        (not (player-at ?from)) (player-at ?to)
        (when (key-tile ?to) (have-key))
        (when (and (key-tile ?to))
              (forall (?c - coord) (when (gate ?c alpha) (gate-open ?c)))
        )
        (when (beacon ?to) (beacon-on ?to))
    )
  )

  ;; west --------------------------------------------------------------
  (:action move-west
    :parameters (?from ?to - coord ?ph - phase)
    :precondition (and
        (player-at ?from) (phase-current ?ph)
        (adj-west ?from ?to)
        (walkable ?to ?ph)
        (not (gate ?to ?ph))
        (not (rift ?to ?ph))
    )
    :effect (and
        (not (player-at ?from)) (player-at ?to)
        (when (key-tile ?to) (have-key))
        (when (and (key-tile ?to))
              (forall (?c - coord) (when (gate ?c alpha) (gate-open ?c)))
        )
        (when (beacon ?to) (beacon-on ?to))
    )
  )

  ;; east --------------------------------------------------------------
  (:action move-east
    :parameters (?from ?to - coord ?ph - phase)
    :precondition (and
        (player-at ?from) (phase-current ?ph)
        (adj-east ?from ?to)
        (walkable ?to ?ph)
        (not (gate ?to ?ph))
        (not (rift ?to ?ph))
    )
    :effect (and
        (not (player-at ?from)) (player-at ?to)
        (when (key-tile ?to) (have-key))
        (when (and (key-tile ?to))
              (forall (?c - coord) (when (gate ?c alpha) (gate-open ?c)))
        )
        (when (beacon ?to) (beacon-on ?to))
    )
  )

  ;; ──────────────────────────────────────────────────────────────────
  ;; PHASE SHIFT  α → β
  ;; ──────────────────────────────────────────────────────────────────
  (:action shift-alpha-beta
    :parameters (?c - coord)
    :precondition (and
        (player-at ?c)
        (phase-current alpha)
        (walkable ?c beta)          ; landing square must exist
        (not (gate ?c beta))        ; cannot phase into closed gate
        (not (rift ?c beta))
    )
    :effect (and
        (not (phase-current alpha))
        (phase-current beta)
    )
  )

  ;; PHASE SHIFT  β → α
  (:action shift-beta-alpha
    :parameters (?c - coord)
    :precondition (and
        (player-at ?c)
        (phase-current beta)
        (walkable ?c alpha)
        (not (gate ?c alpha))
        (not (rift ?c alpha))
    )
    :effect (and
        (not (phase-current beta))
        (phase-current alpha)
    )
  )
)