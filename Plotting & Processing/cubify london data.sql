/****** Script for SelectTopNRows command from SSMS  ******/
SELECT *
FROM [test1].[dbo].[London_301021] x
INNER JOIN Datum d on DATEPART(year, x.[Date]) * 10000 + DATEPART(month, x.[Date]) * 100 + DATEPART(day, x.[Date]) = d.DatumID

exec sp_rename 'test1.dbo.London_301021.transit', 'ttransit'

DROP TABLE IF EXISTS #column_names
SELECT * into #column_names
FROM test1.sys.all_columns where object_id = (SELECT TOP 1 object_id FROM test1.sys.tables where name = 'London_301021')

DECLARE @prefix nvarchar(100) = 'retail_recreation_%'

DROP TABLE IF EXISTS test1.dbo.Borough
SELECT ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) id, RIGHT(name, len(name) - len(@prefix) + 1) name
into test1.dbo.Borough
FROM #column_names
WHERE name like @prefix

IF NOT (SELECT MAX(amount) - MIN(amount)
		FROM (
			--Count how often each kpi exists, i.e. for how many boroughs
			SELECT	--c.name,
					--b.*,
					left(c.name, len(c.name) - (len(b.name) + 1)) kpi_name,
					count(*) amount
			FROM test1.sys.all_columns c
			INNER JOIN test1.dbo.Borough b ON c.name like CONCAT('%', '_', b.name)
			where object_id = (SELECT TOP 1 object_id FROM test1.sys.tables where name = 'London_301021')
			group by left(c.name, len(c.name) - (len(b.name) + 1))
		) kpi
		) = 0
BEGIN
	print 'Inconsistency in kpis per borough. At least one borough has more or less kpis than the others'
	return
END

DROP TABLE IF EXISTS #kpi
SELECT	--c.name,
		--b.*,
		ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) as id,
		left(c.name, len(c.name) - (len(b.name) + 1)) kpi_name
INTO #kpi
FROM test1.sys.all_columns c
INNER JOIN test1.dbo.Borough b ON c.name like CONCAT('%', '_', b.name)
where object_id = (SELECT TOP 1 object_id FROM test1.sys.tables where name = 'London_301021')
group by left(c.name, len(c.name) - (len(b.name) + 1))

DECLARE @kpi_list nvarchar(255)
SELECT @kpi_list = COALESCE(STUFF
	(
		(
			SELECT ', ' + kpi_name
			FROM #kpi
			for xml path('')
		), 1, 2, N''
	), N'')


DECLARE @unpivot_code nvarchar(max) = N''

SELECT @unpivot_code = CONCAT(@unpivot_code, 
N'SELECT date, RIGHT(', (SELECT kpi_name FROM #kpi WHERE id = 1), N'_Borough, charindex(''_'', reverse(', (SELECT kpi_name FROM #kpi WHERE id = 1), N'_Borough) + ''_'') - 1) as Borough, ', @kpi_list,N'
into #unpivoted
FROM (SELECT * FROM London_301021) p
')

DECLARE @i int = 1
DECLARE @kpi_name nvarchar(50)
DECLARE @column_list nvarchar(2000)
WHILE EXISTS (SELECT * FROM #kpi WHERE id = @i)
BEGIN
	SELECT @kpi_name =  kpi_name
	FROM #kpi
	WHERE id = @i

	SELECT @column_list = COALESCE(STUFF
		(
			(
				SELECT N', [' + @kpi_name + '_' + name + ']'
				FROM test1.dbo.Borough
				for xml path('')
			), 1, 2, N''
		), N'')

	SET @unpivot_code = CONCAT(@unpivot_code,
	N'UNPIVOT
		(', @kpi_name, N'
		for ', @kpi_name, N'_Borough in (', @column_list,N')
	) unpiv_', @kpi_name, '
	')

	SET @i = @i + 1
END

IF @i >= 2
BEGIN
	SET @i = 2
	WHILE EXISTS (SELECT * FROM #kpi WHERE id = @i)
	BEGIN
		SET @unpivot_code = CONCAT(@unpivot_code,
		CASE @i 
			WHEN 2 THEN N'WHERE'
			ELSE N'AND'
		END,	N' RIGHT(', (SELECT kpi_name FROM #kpi WHERE id = 1), N'_Borough, charindex(''_'', reverse(', (SELECT kpi_name FROM #kpi WHERE id = 1), N'_Borough) + ''_'') - 1) =',
				N' RIGHT(', (SELECT kpi_name FROM #kpi WHERE id = @i), N'_Borough, charindex(''_'', reverse(', (SELECT kpi_name FROM #kpi WHERE id = @i), N'_Borough) + ''_'') - 1)
				')
		SET @i = @i + 1
	END
END

SET @unpivot_code = CONCAT(@unpivot_code, N'
	SELECT d.DatumID, b.id as BoroughID, ', @kpi_list, N'
	INTO ##fact_table
	FROM #unpivoted u
	INNER JOIN Datum d ON u.date = d.Datum
	INNER JOIN Borough b ON b.name LIKE CONCAT(''%'', u.Borough)
	ORDER BY date
')

SELECT(@unpivot_code)

exec(@unpivot_code)

DROP TABLE IF EXISTS fact_table_modify
SELECT * 
INTO fact_table_modify
FROM ##fact_table
ORDER BY DatumID, BoroughID
