# 6.824: Distributed Systems Engineering


### 2015 Notes 
[Distributed Systems Engineering notes (6.824, Spring 2015)](https://wizardforcel.gitbooks.io/distributed-systems-engineering-lecture-notes/content/)
source [alinush/6.824-lecture-notes](https://github.com/alinush/6.824-lecture-notes)
Based on [6.824 Schedule: Spring 2015](http://nil.csail.mit.edu/6.824/2015/schedule.html)

  
### 2020 Notes
Home Page - [6.824 Home Page: Spring 2020](http://pdos.csail.mit.edu/6.824)
Schedule - [6.824 Schedule: Spring 2020](https://pdos.csail.mit.edu/6.824/schedule.html)
YT Playlist - [MIT 6.824: Distributed Systems](https://www.youtube.com/channel/UC_7WrbZTCODu1o_kfUMq88g)

## Lecture 1: Introduction
From https://pdos.csail.mit.edu/6.824/notes/l01.txt

### What is a distributed system?
*   multiple cooperating computers
*   storage for big websites, MapReduce, peer-to-peer sharing
*   lots of critical infrastructure is distributed

### Why do people build distributed systems?
*   to increase capacity via parallelism
*   to tolerate faults via replication
*   to place computing physically close to external entities
*   to achieve security via isolation

### Why is this hard?
*   many concurrent parts, complex interactions
*   must cope with partial failure
*   tricky to realize performance potential

### Why take this course?
*   interesting -- hard problems, powerful solutions
*   used by real systems -- driven by the rise of big websites
*   active research area -- important unsolved problems
*   hands-on -- you'll build real systems in the labs


### Course Components
Lectures
*   big ideas, paper discussion, and labs
*   will be video-taped, available online
Papers
*   research papers, some classic, some new
*   problems, ideas, implementation details, evaluation
*   many lectures focus on papers
*   please read papers before class!
*   each paper has a short question for you to answer
*   and we ask you to send us a question you have about the paper
*   submit question & answer by midnight the night before

Labs
* Lab 1: MapReduce
* Lab 2: replication for fault-tolerance using Raft
* Lab 3: fault-tolerant key / value store
* Lab 4: sharded key / value store
Course Material
This is a course about infrastructure for applications
* Storage
* Communication
* Computation
Abstractions
*  that hide the complexity of distributed systems.
Implementation
  RPC, threads, concurrency control
Performance
*   The goal: scalable throughput (horizontally scalable) 
   * N servers -> N x total throughput via parallel CPU, disk, net
   * [diagram: users, application servers, storage servers]
*     So handling more load only requires buying more computers
   * rather than redesign by expensive programmers
   * effective when you can divide work w/o much interaction
*   Scaling gets harder as N grows
   * Load im-balance, stragglers, slowest-of-N latency (tail latencies)
   * Non-parallelizable code: initialization, interaction
   * Bottlenecks from shared resources, e.g. network
*   Some performance problems aren't easily solved by (horizontal) scaling
   * e.g. quick response time for a single user request
   * e.g. all users want to update the same data
   * often requires better design rather than just more computers
Fault Tolerance
* 1000s of servers, big network -> always something broken
*  We'd like to hide these failures from the application
*  We often want -
   *  Availability -- app can make progress despite failures (stronger)
   *  Recoverability -- app will come back to life when failures are repaired (weaker)
*  Big idea: replicated servers
   *  If one server crashes, can proceed using the other(s)
* Non volatile storage 
   * Checkpoint to recover
Consistency
* General-purpose infrastructure needs well-defined behavior.
   * E.g. "Get(k) yields the value from the most recent Put(k,v)."
* Achieving good behavior is hard!
   * "Replica" servers are hard to keep identical
   * Clients may crash midway through a multi-step update
   * Servers may crash, e.g. after executing but before replying
   * Network partition may make live servers look dead; risk of "split brain".
*   Consistency and performance are enemies.
   * Strong consistency requires communication
   * e.g. Get() must check for a recent Put().
*   Many designs provide only weak consistency, to gain speed.
   * e.g. Get() does *not* yield the latest Put()!
   * Painful for application programmers but may be a good trade-off.
*   Many design points are possible in the consistency/performance spectrum!

## CASE STUDY: MapReduce

### Overview
*   context: multi-hour computations on multi-terabyte data-sets
   * e.g. build search index, or sort, or analyze structure of web
   * only practical with 1000s of computers
   * applications not written by distributed systems experts
*   overall goal: easy for non-specialist programmers
*   programmer just defines Map and Reduce functions
   * often fairly simple sequential code
*   MR takes care of, and hides, all aspects of distribution!
Abstract View
Input is (already) split into M files
  Input1 -> Map ->  a,1 b,1
  Input2 -> Map ->        b,1
  Input3 -> Map ->  a,1      c,1
                    |   |   |
                    |   |   -> Reduce -> c,1
                    |   -----> Reduce -> b,2
                    ---------> Reduce -> a,2
  
* MR calls Map() for each input file, produces set of k2,v2
   * "intermediate" data
   * each Map() call is a "task"
*   MR gathers all intermediate v2's for a given k2,
   * and passes each key + values to a Reduce call
*   final output is set of <k2,v3> pairs from Reduce()s
Example: Word Count


Input is thousands of text files


Map(k, v):
    split v into words
    for each word w:
        emit(w, "1")


Reduce(k, v):
    emit(len(v))
	

  

Scaling
*   N "worker" computers get you N x throughput.
   * Maps()s can run in parallel, since they don't interact.
   * Same for Reduce()s.
*   So you can get more throughput by buying more computers.
Encapsulation
*   sending app code to servers
*   tracking which tasks are done
*   moving data from Maps to Reduces
*   balancing load over servers
*   recovering from failures
Limitation 
*   No interaction or state (other than via intermediate output).
*   No iteration, no multi-stage pipelines.
*   No real-time or streaming processing.
Storage on the GFS cluster file system
*   MR needs huge parallel input and output throughput.
*   GFS splits files over many servers, in 64 MB chunks
   * Maps read in parallel
   * Reduces write in parallel
*   GFS also replicates each file on 2 or 3 servers
*   Having GFS is a big win for MapReduce
Performance Limitation
*  CPU? memory? disk? network?
*  In 2004 authors were limited by network capacity.
*  What does MR send over the network?
      * Maps read input from GFS.
      * Reduces read Map output.
         * Can be as large as input, e.g. for sorting.
      * Reduces write output files to GFS
[diagram: servers, tree of network switches]
*     In MR's all-to-all shuffle, half of traffic goes through the root switch.
*     Paper's root switch: 100 to 200 gigabits/second, total
   * 1800 machines, so 55 megabits/second/machine.
   * 55 is small, e.g. much less than disk or RAM speed.
*   Today: networks and root switches are much faster relative to CPU/disk.
MR Execution Details
  

Figure 1: Execution overview (from paper)


1. One master, that hands out tasks to workers and remembers progress.
2. Master gives Map tasks to workers until all Maps complete
3. Maps write output (intermediate data) to local disk
4. Maps split output, by hash, into one file per Reduce task
5. After all Maps have finished, master hands out Reduce tasks
6. Each Reduce fetches its intermediate output from (all) Map workers
7. Each Reduce task writes a separate output file on GFS
MR Network Usage
* Master tries to run each Map task on GFS server that stores its input.[a]
   * All computers run both GFS and MR workers
   * So input is read from local disk (via GFS), not over network.
*   Map workers write to the local disk.
   *   Intermediate (From Mappers to Reducer) data goes over the network just once.
   *  Reduce workers read directly from Map workers, not via GFS.
*   Intermediate data partitioned into files holding many keys.
*     R is much smaller than the number of keys.
*     Big network transfers are more efficient.


How does MR get good load balance?
  Wasteful and slow if N-1 servers have to wait for 1 slow server to finish.
  But some tasks likely take longer than others.
  Solution: many more tasks than workers.
    Master hands out new tasks to workers who finish previous tasks.
    So no task is so big it dominates completion time (hopefully).
    So faster servers do more tasks than slower ones, finish abt the same time.


MR Fault Tolerance
  I.e. what if a worker crashes during a MR job?
  We want to completely hide failures from the application programmer!
  Does MR have to re-run the whole job from the beginning?
    Why not?
  MR re-runs just the failed Map()s and Reduce()s.
    Suppose MR runs a Map twice, one Reduce sees first run's output,
      another Reduce sees the second run's output?
    Correctness requires re-execution to yield exactly the same output.
    So Map and Reduce must be pure deterministic functions:
      they are only allowed to look at their arguments.
      no state, no file I/O, no interaction, no external communication.
  What if you wanted to allow non-functional Map or Reduce?
    Worker failure would require whole job to be re-executed,
      or you'd need to create synchronized global checkpoints.


Details of worker crash recovery:
  * Map worker crashes:
    master notices worker no longer responds to pings
    master knows which Map tasks it ran on that worker
      those tasks' intermediate output is now lost, must be re-created
      master tells other workers to run those tasks
    can omit re-running if Reduces already fetched the intermediate data
  * Reduce worker crashes.
    finished tasks are OK -- stored in GFS, with replicas.
    master re-starts worker's unfinished tasks on other workers.


Other failures/problems:
  * What if the master gives two workers the same Map() task?
    perhaps the master incorrectly thinks one worker died.
    it will tell Reduce workers about only one of them.
  * What if the master gives two workers the same Reduce() task?
    they will both try to write the same output file on GFS!
    atomic GFS rename prevents mixing; one complete file will be visible.
  * What if a single worker is very slow -- a "straggler"?
    perhaps due to flakey hardware.
    master starts a second copy of last few tasks.
  * What if a worker computes incorrect output, due to broken h/w or s/w?
    too bad! MR assumes "fail-stop" CPUs and software.
  * What if the master crashes?


Current status?
  Hugely influential (Hadoop, Spark, &c).
  Probably no longer in use at Google.
    Replaced by Flume / FlumeJava (see paper by Chambers et al).
    GFS replaced Colossus (no good description), and BigTable.


### Conclusion
  MapReduce single-handedly made big cluster computation popular.
  - Not the most efficient or flexible.
  + Scales well.
  + Easy to program -- failures and data movement are hidden.
  These were good trade-offs in practice.
  We'll see some more advanced successors later in the course.
  Have fun with the lab!


[a]This is not true
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5NzMyMzg4MTEsMTM2MjM1NTkwNF19
-->