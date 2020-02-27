(define (problem strips-gripper-x-2)
   (:domain gripper-strips)
   (:objects rooma roomb roomc roomd ball4 ball3 ball2 ball1 left1 right1 rob1 left2 right2)
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
          (free left2)
          (free right2)

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
