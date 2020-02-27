(define (problem strips-gripper-x-2)
   (:domain gripper-strips)
   (:objects rooma roomb ball1 ball2 ball3 ball4 ball5 ball6 left1 rob1)
   (:init (room rooma)
          (room roomb)

          (ball ball1)
          (ball ball2)
          (ball ball3)
          (ball ball4)
          (ball ball5)
          (ball ball6)

          (robot rob1)

          (at-robby rob1 rooma)

          (free left1)


          (at ball1 rooma)
          (at ball2 rooma)
          (at ball3 rooma)
          (at ball4 rooma)
          (at ball5 rooma)
          (at ball6 rooma)

          (gripper left1 rob1)
          )

   (:goal (and
               (at ball1 roomb)
               (at ball2 roomb)
               (at ball3 roomb)
               (at ball4 roomb)
               (at ball5 roomb)
               (at ball6 roomb))))
