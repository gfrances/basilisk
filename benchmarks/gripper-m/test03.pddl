(define (problem strips-gripper-x-2)
   (:domain gripper-strips)
   (:objects rooma roomb roomc ball2 ball1 left1 right1 rob1)
   (:init (room rooma)
          (room roomb)
          (room roomc)

          (ball ball2)
          (ball ball1)

          (robot rob1)

          (at-robby rob1 rooma)

          (free left1)
          (free right1)

          (at ball2 roomc)
          (at ball1 rooma)

          (gripper left1 rob1)
          (gripper right1 rob1)
          )

   (:goal (and
               (at ball2 roomb)
               (at ball1 roomb))))
