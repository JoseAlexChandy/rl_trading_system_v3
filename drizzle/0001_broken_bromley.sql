CREATE TABLE `agent_logs` (
	`id` int AUTO_INCREMENT NOT NULL,
	`level` enum('info','warn','error','trade','risk') NOT NULL DEFAULT 'info',
	`message` text NOT NULL,
	`metadata` json,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `agent_logs_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `model_versions` (
	`id` int AUTO_INCREMENT NOT NULL,
	`version` varchar(64) NOT NULL,
	`trainingRunId` int,
	`isActive` boolean NOT NULL DEFAULT false,
	`s3Path` text NOT NULL,
	`metrics` json,
	`deployedAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `model_versions_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `portfolio_snapshots` (
	`id` int AUTO_INCREMENT NOT NULL,
	`totalValue` float NOT NULL,
	`cash` float NOT NULL,
	`unrealizedPnl` float NOT NULL,
	`realizedPnl` float NOT NULL,
	`drawdown` float NOT NULL,
	`sharpeRatio` float,
	`winRate` float,
	`totalTrades` int NOT NULL,
	`openPositions` int NOT NULL,
	`snapshotAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `portfolio_snapshots_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `trades` (
	`id` int AUTO_INCREMENT NOT NULL,
	`asset` varchar(32) NOT NULL,
	`side` enum('long','short') NOT NULL,
	`entryPrice` float NOT NULL,
	`exitPrice` float,
	`quantity` float NOT NULL,
	`leverage` float NOT NULL,
	`pnl` float,
	`pnlPercent` float,
	`status` enum('open','closed','liquidated') NOT NULL DEFAULT 'open',
	`mode` enum('swing','scalp') NOT NULL DEFAULT 'swing',
	`confidence` float,
	`sentimentScore` float,
	`entryReason` text,
	`exitReason` text,
	`openedAt` timestamp NOT NULL DEFAULT (now()),
	`closedAt` timestamp,
	CONSTRAINT `trades_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `training_runs` (
	`id` int AUTO_INCREMENT NOT NULL,
	`runName` varchar(128) NOT NULL,
	`status` enum('running','completed','failed','paused') NOT NULL DEFAULT 'running',
	`mode` enum('offline','online') NOT NULL DEFAULT 'offline',
	`totalEpisodes` int NOT NULL,
	`completedEpisodes` int NOT NULL DEFAULT 0,
	`bestReward` float,
	`avgReward` float,
	`finalPortfolioValue` float,
	`hyperparameters` json,
	`s3ModelPath` text,
	`s3BufferPath` text,
	`startedAt` timestamp NOT NULL DEFAULT (now()),
	`completedAt` timestamp,
	CONSTRAINT `training_runs_id` PRIMARY KEY(`id`)
);
