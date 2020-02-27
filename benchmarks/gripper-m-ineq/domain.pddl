(define (domain gripper-strips)
   (:predicates (room ?r)
		(robot ?r)
		(ball ?b)
		(gripper ?g ?r)
		(at-robby ?rob ?room)
		(at ?b ?r)
		(free ?g)
		(carry ?o ?g))

   (:action move
       :parameters  (?r ?from ?to)
       :precondition (and  (room ?from) (room ?to) (at-robby ?r ?from) (not (= ?from ?to)))
       :effect (and  (at-robby ?r ?to)
		     (not (at-robby ?r ?from))))



   (:action pick
       :parameters (?obj ?room ?rob ?gripper)
       :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper ?rob)
			    (at ?obj ?room) (at-robby ?rob ?room) (free ?gripper))
       :effect (and (carry ?obj ?gripper)
		    (not (at ?obj ?room))
		    (not (free ?gripper))))


   (:action drop
       :parameters  (?obj  ?room ?rob ?gripper)
       :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper ?rob)
			    (carry ?obj ?gripper) (at-robby ?rob ?room))
       :effect (and (at ?obj ?room)
		    (free ?gripper)
		    (not (carry ?obj ?gripper)))))
