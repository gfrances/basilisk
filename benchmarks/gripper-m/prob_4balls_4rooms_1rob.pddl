(define (problem strips-gripper-x-2)
   (:domain gripper-strips)
   (:objects rooma roomb roomc roomd ball3 ball2 ball1 left1 right1 rob1)
   (:init (room rooma)
          (room roomb)
          (room roomc)
          (room roomd)

          (ball ball4)
          (ball ball3)
          (ball ball2)
          (ball ball1)

          (robot rob1)

          (at-robby rob1 roomd)

          (free left1)
          (free right1)

          (at ball4 roomd)
          (at ball3 roomb)
          (at ball2 rooma)
          (at ball1 roomc)

          (gripper left1 rob1)
          (gripper right1 rob1)
          )

   (:goal (and
               (at ball4 roomb)
               (at ball3 roomb)
               (at ball2 roomb)
               (at ball1 roomb))))
